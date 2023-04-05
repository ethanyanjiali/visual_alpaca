import csv
import tempfile
import concurrent.futures
import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.io import parquetio
from apache_beam.options.pipeline_options import PipelineOptions
import click
import os
import gc
import requests
import logging
import time
from io import BytesIO
from PIL import Image
import webdataset as wds
import urllib3
import uuid
from google.cloud import storage


def download_process_write_image(element, image_size):
    url = element['URL']
    try:
        content = download_with_retry(url)
    except Exception as e:
        # Log the error and skip the element
        logging.error(f'Error downloading image from URL {url}: {str(e)}')
        return

    try:
        content = decode_resize_image(content, image_size)
    except Exception as e:
        # Log the error and skip the element
        logging.error(f'Error decoding and resizing the image from URL {url}: {str(e)}')
        return

    return content, element


def decode_resize_image(content, image_size):
    image = Image.open(BytesIO(content))
    image.thumbnail((image_size, image_size))
    image = image.convert("RGB")
    image_bytes = BytesIO()
    image.save(image_bytes, 'JPEG')
    return image_bytes.getvalue()


def download_with_retry(url, max_retries=1):
    for i in range(max_retries):
        try:
            response = requests.get(url, verify=False, timeout=5)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            if i == max_retries - 1:
                raise e
            else:
                time.sleep(1)


class DownloadAndWriteToWds(beam.DoFn):
    def __init__(self, output_dir, image_size=384, num_threads=12):
        # https://issues.apache.org/jira/browse/BEAM-6158
        self.output_dir = output_dir
        self.image_size = image_size
        self.num_threads = num_threads
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def process(self, elements):
        client = storage.Client()
        bucket = client.get_bucket(self.output_dir.split("/")[2])

        logging.info(f'Start processing {len(elements)} elements.')
        outputs = []
        # our download is mostly I/O bound
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for element in elements:
                future = executor.submit(download_process_write_image, element, self.image_size)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    outputs.append(result)

        # Write files into Web dataset format
        tar_id = str(uuid.uuid4())
        tar_name = f'{tar_id}.tar'
        tar_path = os.path.join(self.output_dir, tar_name)
        ids_path = os.path.join(self.output_dir, f'{tar_id}.txt')

        max_retries = 3
        with tempfile.NamedTemporaryFile('wb') as f1, tempfile.NamedTemporaryFile('w') as f2:
            csv_writer = csv.writer(f2)
            with wds.TarWriter(f1) as tar_writer:
                for output in outputs:
                    content = output[0]
                    element = output[1]
                    sample_id = element['SAMPLE_ID']
                    sample = {
                        '__key__': str(sample_id),
                        'jpg': content,
                        'txt': element['TEXT'] if element['TEXT'] is not None else "",
                    }
                    tar_writer.write(sample)
                    csv_writer.writerow([sample_id, tar_name])

            f1.flush()
            f2.flush()

            for i in range(max_retries):
                try:
                    blob = bucket.blob('/'.join(tar_path.split('/')[3:]))
                    blob.upload_from_filename(f1.name)
                except Exception as e:
                    if i == max_retries - 1:
                        logging.error(
                            f'Unable to copy tarfile to {tar_path}: {str(e)}')
                    else:
                        time.sleep(1)

            for i in range(max_retries):
                try:
                    blob = bucket.blob('/'.join(ids_path.split('/')[3:]))
                    blob.upload_from_filename(f2.name)
                except Exception as e:
                    if i == max_retries - 1:
                        logging.error(
                            f'Unable to copy to tar ids file {ids_path}: {str(e)}')
                    else:
                        time.sleep(1)

        logging.info(f'Finished processing {len(outputs)} out of {len(elements)} elements.')
        del outputs
        gc.collect()


@click.command()
@click.option('--parquet_dir', type=str, required=True)
@click.option('--runner', default='DataflowRunner', type=click.Choice(['DataflowRunner', 'DirectRunner']))
@click.option('--project', type=str, required=True)
@click.option('--region', type=str, required=True)
@click.option('--job_name', type=str, required=True)
@click.option('--wds_dir', type=str, required=True)
@click.option('--temp_location', type=str, required=True)
@click.option('--image_size', type=int, default=384)
@click.option('--tar_size', type=int, default=10000)
@click.option('--max_workers', type=int, default=24)
@click.option('--num_threads', type=int, default=12)
@click.option('--machine_type', type=str, default='n1-standard-1')
def main(parquet_dir, runner, project, region, job_name, wds_dir, temp_location, image_size, tar_size, max_workers,
         num_threads, machine_type):
    beam_options = PipelineOptions(
        runner=runner,
        project=project,
        region=region,
        job_name=job_name,
        temp_location=temp_location,
        max_num_workers=max_workers,
        wait_until_finish=False,
        requirements_file="requirements.in",
        save_main_session=True,
        # https://issues.apache.org/jira/browse/BEAM-4112
        machine_type=machine_type,
    )

    try:
        FileSystems.mkdirs(wds_dir)
    except Exception as e:
        logging.error(f'Unable to create directory {wds_dir}. It may already exist. ' + str(e))

    pipeline = beam.Pipeline(options=beam_options)
    # Read all parquet files from the GCS directory
    rows = pipeline | 'Read Parquet Files' >> parquetio.ReadFromParquet(
        file_pattern=os.path.join(parquet_dir, '*.parquet'))
    batches = rows | 'Batch Samples' >> beam.BatchElements(min_batch_size=tar_size,
                                                           max_batch_size=tar_size)

    batches | 'Download and Write WDS' >> beam.ParDo(
        DownloadAndWriteToWds(output_dir=wds_dir, image_size=image_size, num_threads=num_threads))

    pipeline.run()


if __name__ == '__main__':
    main()
