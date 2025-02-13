import boto3
import traceback
from datetime import datetime
import uuid
import os
import glob
import json
import time
import argparse
from pathlib import Path

from typing import Union

from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAIEmbeddingService, OpenAILLMService

DOMAIN = "Analyze this TypeScript code and identify the components, functions, and their relationships."

EXAMPLE_QUERIES = [
    "What are the main components in this codebase?",
    "How do different functions interact with each other?",
    "What are the key interfaces and types defined?",
    "Describe the main functionality of this codebase.",
    "What are the dependencies between different files?",
]

ENTITY_TYPES = ["Component", "Function", "Interface", "Type", "Variable", "Class"]

session = boto3.session.Session(profile_name="PROFILE_NAME")
bedrock_client = session.client(service_name="bedrock", region_name="us-west-2")
s3_client = session.client(service_name="s3", region_name="us-west-2")

S3_BUCKET = os.environ["S3_BUCKET"]
HAIKU_MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
SONNET_MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
QWEN_MODEL_ID = "qwen2.5-coder:7b"
SERVICE_ROLE = os.environ["SERVICE_ROLE"]

BEDROCK_BATCH_MIN_PROMPTS = 100
BEDROCK_BATCH_SIZE = 5000
BEDROCK_BATCH_MAX_PROMPTS = 50000


# TODO: handle multiple input paths
def gather_files(directory_path, extensions):
    output = []
    files = sum(
        [glob.glob(os.path.join(directory_path, f"**/*.{ext}"), recursive=True) for ext in extensions],
        [],
    )

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                file_content = f"// <filepath>{file_path}</filepath>\n{content}"
                output.append(file_content)
        except Exception as e:
            print(f"[gather_files] Error processing file {file_path}: {e}")

    return output


# TODO: batch this
def insert_files(directory_path, extensions, grag, max_retries=3, backoff_base=2):
    files = sum(
        [glob.glob(os.path.join(directory_path, f"**/*.{ext}"), recursive=True) for ext in extensions],
        [],
    )

    total_files = len(files)
    processed_files = 0
    failed_files = []

    print(f"Starting processing of {total_files} files...")

    # First pass through files
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                file_content = f"// <filepath>{file_path}</filepath>\n\n{content}"
                grag.insert(file_content)
                processed_files += 1
                print(f"Processed {processed_files}/{total_files} files ({(processed_files / total_files) * 100:.1f}%)")
        except Exception as e:
            print(f"[insert_files] Error processing file {file_path}: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            failed_files.append((file_path, 1))
            processed_files += 1
            print(
                f"Processed {processed_files}/{total_files} files ({(processed_files / total_files) * 100:.1f}%) - with error"
            )

    # Retry failed files with exponential backoff
    time.sleep(10 * 60)
    os.environ["CONCURRENT_TASK_LIMIT"] = int(os.environ["CONCURRENT_TASK_LIMIT"]) // 2
    while failed_files:
        still_failed = []
        print(f"\nRetrying {len(failed_files)} failed files...")

        for file_path, attempts in failed_files:
            if attempts > max_retries:
                print(f"[insert_files] Permanently failed to process {file_path} after {max_retries} attempts")
                continue

            # Wait with exponential backoff
            wait_time = backoff_base**attempts
            print(f"[insert_files] Retrying {file_path} (attempt {attempts}) after {wait_time}s delay")
            time.sleep(wait_time)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    file_content = f"// <filepath>{file_path}</filepath>\n\n{content}"
                    grag.insert(file_content)
                    print(f"[insert_files] Successfully processed {file_path} on retry")
            except Exception as e:
                print(f"[insert_files] Error processing file {file_path} on retry: {e}")
                still_failed.append((file_path, attempts + 1))

        failed_files = still_failed
        if failed_files:
            print(f"[insert_files] {len(failed_files)} files still failing, continuing retries...")

    print(f"\nCompleted processing {total_files} files with {len(failed_files)} permanent failures")


def interactive_questions(grag):
    print("\nEnter your questions (type 'quit' or 'exit' to end):")
    while True:
        try:
            question = input("\nQuestion: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                break
            if not question:
                continue

            response = grag.query(question).response
            print("\nResponse:")
            print(response)
            print("\n" + "-" * 80)  # separator line

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


def ask_question(grag, question):
    print(f"Question: {question}")
    response = grag.query(question).response
    print(f"Response: {response}")
    return response


def create_unique_job_name(prefix: str = "bedrock-batch", max_length: int = 64) -> str:
    """
    Create a unique job name using timestamp and random string

    Args:
        prefix: Prefix for the job name
        max_length: Maximum length for the job name (Bedrock limit)

    Returns:
        str: Unique job name
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    # Take first 8 characters of a UUID
    random_suffix = str(uuid.uuid4())[:8]

    # Combine parts and ensure we don't exceed max length
    job_name = f"{prefix}-{timestamp}-{random_suffix}"
    if len(job_name) > max_length:
        # If too long, truncate the prefix
        available_length = max_length - len(f"-{timestamp}-{random_suffix}")
        job_name = f"{prefix[:available_length]}-{timestamp}-{random_suffix}"

    return job_name


def split_file_into_batches(file_path: Path, min_lines: int, max_lines: int) -> list[list[str]]:
    """
    Split a file into batches based on number of lines

    Args:
        file_path: Path to the input file
        min_lines: Minimum number of lines per batch
        max_lines: Maximum number of lines per batch

    Returns:
        List of batches, where each batch is a list of lines
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    total_lines = len(lines)
    batches = []

    # If total lines is less than max_lines, just return one batch
    if total_lines <= max_lines:
        return [lines]

    # Calculate number of batches needed
    num_full_batches = total_lines // max_lines
    remainder = total_lines % max_lines

    # Create full-sized batches
    for i in range(num_full_batches):
        start_idx = i * max_lines
        end_idx = start_idx + max_lines
        batches.append(lines[start_idx:end_idx])

    # Handle remainder
    if remainder:
        if remainder < min_lines and batches:
            # If remainder is too small, append to last batch
            batches[-1].extend(lines[num_full_batches * max_lines :])
        else:
            # Otherwise make it its own batch
            batches.append(lines[num_full_batches * max_lines :])

    print(f"{file_path} has length {total_lines}, creating {len(batches)} jobs")
    return batches


def upload_to_s3(file_path: Union[str, Path], bucket: str, key: str = None) -> bool:
    """
    Upload a file to S3

    Args:
        file_path: Local path to file to upload
        bucket: S3 bucket name
        key: S3 key (path) to upload to. If None, uses filename

    Returns:
        bool: True if successful, False if failed
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False

    if key is None:
        key = file_path.name

    try:
        s3_client.upload_file(str(file_path), bucket, key)
        print(f"Successfully uploaded {file_path} to s3://{bucket}/{key}")
        return True
    except Exception as e:
        print(f"Failed to upload file: {e}")
        return False


def create_bedrock_jobs(base_path: Path, file_name: str, job_name: str, model_id: str) -> list[tuple[int, str]]:
    """ """
    file_path = base_path / file_name
    batches = split_file_into_batches(file_path, BEDROCK_BATCH_MIN_PROMPTS, BEDROCK_BATCH_SIZE)
    job_info = []

    # Split the filename and extension
    name_parts = file_name.rsplit(".", 1)
    base_name = name_parts[0]
    extension = name_parts[1] if len(name_parts) > 1 else ""

    for i, batch in enumerate(batches):
        if len(batches) > 1:
            batch_file_name = f"{base_name}.batch{i}.{extension}"
            batch_path = base_path / batch_file_name

            with open(batch_path, "w") as f:
                f.writelines(batch)
        else:
            batch_file_name = f"{base_name}.{extension}"
            batch_path = base_path / batch_file_name

        # Upload and create job
        upload_to_s3(batch_path, S3_BUCKET)

        input_data_config = {"s3InputDataConfig": {"s3Uri": f"s3://{S3_BUCKET}/{batch_file_name}"}}
        output_data_config = {"s3OutputDataConfig": {"s3Uri": f"s3://{S3_BUCKET}/claude-output/"}}

        response = bedrock_client.create_model_invocation_job(
            roleArn=SERVICE_ROLE,
            modelId=model_id,
            jobName=create_unique_job_name(f"{job_name}-batch{i}"),
            inputDataConfig=input_data_config,
            outputDataConfig=output_data_config,
        )

        job_info.append(response.get("jobArn"))

        # Clean up temporary batch file
        # batch_path.unlink()

    return job_info


def insert_batch_number(s3_key: str, batch_num: int) -> str:
    # Split at the last occurrence of '/'
    path, filename = s3_key.rsplit("/", 1)
    # Split filename at first '.'
    base_name, rest = filename.split(".", 1)
    return f"{path}/{base_name}.batch{batch_num}.{rest}"


def wait_for_batch_jobs(
    job_arns: list[str],
    output_dir: Path,
    output_file_name: str,
    poll_interval_seconds: int = 30,
) -> bool:
    """
    Wait for all batch jobs to complete and combine their outputs
    """
    output_file = output_dir / f"{output_file_name}"
    if output_file.exists():
        return True

    completed_jobs = set()
    temp_files = {}

    while len(completed_jobs) < len(job_arns):
        for i, job_arn in enumerate(job_arns):
            if job_arn in completed_jobs:
                continue

            response = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
            status = response["status"]

            if status in ["Completed", "PartiallyCompleted"]:
                print(f"Job {job_arn} completed")

                # Download output
                output_s3_uri = response["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
                s3_parts = output_s3_uri.replace("s3://", "").split("/")
                bucket = s3_parts[0]
                folder_name = job_arn.split("/")[-1]
                key = "/".join(
                    [
                        "claude-output",
                        folder_name,
                        output_file_name[: output_file_name.rindex(".")],
                    ]
                )
                if len(job_arns) > 1:
                    key = insert_batch_number(key, i)

                if len(job_arns) > 1:
                    temp_file = output_dir / f"{output_file_name}.batch{i}"
                else:
                    temp_file = output_dir / f"{output_file_name}"
                s3_client.download_file(Bucket=bucket, Key=key, Filename=str(temp_file))
                temp_files[job_arn] = temp_file
                completed_jobs.add(job_arn)

            elif status in ["Expired", "Failed", "Stopped"]:
                print(f"Job {job_arn} failed: {response.get('failureReason', 'Unknown error')}")
                return False

        if len(completed_jobs) < len(job_arns):
            print(f"Waiting for {len(job_arns) - len(completed_jobs)} jobs to complete...")
            time.sleep(poll_interval_seconds)

    if len(job_arns) > 1:
        # Combine output files in correct order
        with open(output_file, "w") as outfile:
            for job in job_arns:
                temp_file_path = temp_files[job]
                with open(temp_file_path, "r") as infile:
                    outfile.write(infile.read())
                # Call unlink on the Path object, not the file object
                temp_file_path.unlink()  # Clean up temp files

    return True


def read_file(path):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = f.read()
            return data if data else None
    except:
        return None


def read_file_lines(path):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            lines = f.readlines()
            return lines if lines else None
    except:
        return None


class JobsManager:
    def __init__(self, source_dir: Path, work_dir: Path, base_path: Path, model_id: str):
        self.job_arns_file = Path(work_dir) / "jobArns.json"
        self.source_dir = source_dir
        self.work_dir = work_dir
        self.base_path = base_path
        self.model_id = model_id

        if not self.job_arns_file.exists():
            with open(self.job_arns_file, "w") as f:
                json.dump({}, f)

        self.job_file = json.loads(read_file(self.job_arns_file))
        self.arns = self.job_file.setdefault(self.source_dir, {})

    def update_arns(self, task_name: str, arns: list):
        self.arns[task_name] = arns
        self.job_file[self.source_dir] = self.arns
        write_file(self.job_arns_file, json.dumps(self.job_file, indent=2))

    def get_or_create(self, task_name: str, prompt_file_name: str, callback=None):
        if not self.arns.get(task_name):
            if callback:
                callback()
            job_arns = create_bedrock_jobs(self.base_path, prompt_file_name, task_name, self.model_id)
            self.update_arns(task_name, job_arns)
        arns = self.arns[task_name]
        file_out = f"{prompt_file_name}.out.{arns[0].split('/')[-1]}"
        success = self.wait_for(arns, file_out)
        if success:
            return read_file_lines(self.base_path / file_out)

    def wait_for(self, arns: list, out_file_name: str, poll_interval_secs: int = 300):
        return wait_for_batch_jobs(arns, self.base_path, out_file_name, poll_interval_secs)


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)


def get_llm_config(llm_choice):
    if llm_choice == "qwen":
        return {
            "model": QWEN_MODEL_ID,
            "base_url": "http://localhost:11434/v1/",
            "client": "openai",
        }
    elif llm_choice == "sonnet":
        return {
            "model": SONNET_MODEL_ID,
            "base_url": "http://localhost:8000/",
            "client": "anthropic",
        }
    else:  # haiku
        return {
            "model": HAIKU_MODEL_ID,
            "base_url": "http://localhost:8000/",
            "client": "anthropic",
        }


def main():
    parser = argparse.ArgumentParser(description="Create knowledge graph for LLM RAG")
    parser.add_argument("--path", required=True, type=str, help="Directory to source files from")
    parser.add_argument("--work_dir", required=True, type=str, help="Directory to store computed data")
    parser.add_argument(
        "--query",
        action=argparse.BooleanOptionalAction,
        help="Whether to enter interactive query mode",
    )
    parser.add_argument(
        "--batch",
        action=argparse.BooleanOptionalAction,
        help="Whether to build using batch mode",
    )
    parser.add_argument(
        "--build",
        action=argparse.BooleanOptionalAction,
        help="Whether to build",
    )
    parser.add_argument(
        "--llm",
        choices=["qwen", "sonnet", "haiku"],
        default="qwen",
        help="Select LLM service to use (qwen, sonnet, or haiku)",
    )
    args = parser.parse_args()

    source_directory = args.path

    llm_config = get_llm_config(args.llm)
    config = GraphRAG.Config(
        llm_service=OpenAILLMService(api_key="bedrock", **llm_config),
        embedding_service=OpenAIEmbeddingService(
            model="cohere.embed-english-v3",
            base_url="http://localhost:8000/api/v1/",
            api_key="bedrock",
            embedding_dim=1024,
        ),
    )

    # TODO: make BatchGraphRAG variant / Bedrock batch_service
    grag = GraphRAG(
        working_dir=args.work_dir,
        domain=DOMAIN,
        example_queries="\n".join(EXAMPLE_QUERIES),
        entity_types=ENTITY_TYPES,
        config=config,
    )

    base_path = Path(args.work_dir) / "batch_prompts"
    base_path.mkdir(parents=True, exist_ok=True)

    jobs_manager = JobsManager(source_directory, args.work_dir, base_path, llm_config.model)

    extensions = ["ts", "tsx", "json", "yaml", "md", "js"]
    if args.build:
        print(f"Beginning {'batch ' if args.batch else ''}job for: {source_directory}")

        if not args.batch:
            insert_files(source_directory, extensions, grag)
        else:
            file_contents = gather_files(source_directory, extensions)
            extraction_prompt_file_name = "entity_relationship_extraction.jsonl"
            glean_prompt_file_name = "entity_relationship_glean.jsonl"
            summarize_nodes_prompt_file_name = "summarize_nodes_description.jsonl"
            summarize_edges_prompt_file_name = "summarize_edges_description.jsonl"

            (chunks, succeeded) = grag.prepare_batch_extraction_prompt(
                file_contents, base_path / extraction_prompt_file_name
            )

            extract_output = jobs_manager.get_or_create("batch-extract", extraction_prompt_file_name)
            if not extract_output:
                return

            glean_output = jobs_manager.get_or_create(
                "glean-extract",
                glean_prompt_file_name,
                callback=lambda: grag.prepare_batch_glean_prompt(
                    chunks, base_path / glean_prompt_file_name, extract_output
                ),
            )
            if not glean_output:
                return

            subgraphs = grag.batch_insert(chunks, extract_output, glean_output)
            # TODO: handle updates to existing graph
            graphs = grag.batch_generate_graphs(subgraphs, chunks, Path(args.work_dir) / "graphs_cache.pkl")

            summarize_nodes_output = jobs_manager.get_or_create(
                "summarize-nodes-description",
                summarize_nodes_prompt_file_name,
                lambda: grag.prepare_batch_node_summaries(graphs, base_path / summarize_nodes_prompt_file_name),
            )
            if not summarize_nodes_output:
                return

            upserted_nodes = grag.prepare_batch_edge_summaries(
                graphs,
                summarize_nodes_output,
                base_path / summarize_edges_prompt_file_name,
                Path(args.work_dir) / "upserted_nodes_cache.pkl",
            )

            summarize_edges_output = jobs_manager.get_or_create(
                "summarize-edges-description",
                summarize_edges_prompt_file_name,
            )
            if not summarize_edges_output:
                return

            grag.batch_upsert(
                chunks,
                graphs,
                upserted_nodes,
                summarize_edges_output,
            )

    if args.query:
        ask_question(grag, "What are the main components in this codebase?")
        interactive_questions(grag)
        return


if __name__ == "__main__":
    main()
