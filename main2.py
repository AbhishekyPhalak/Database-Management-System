import heapq
import json
import hashlib
import re
import os
import shutil
import sys
from pathlib import Path
import tempfile


# Make sure 'database' directory exists
if not os.path.isdir('database'):
    os.mkdir('database')


def get_nested_value(nested_keys, data_dict):
    for key in nested_keys:
        data_dict = data_dict.get(key, {})
    return data_dict

def segregate_data(filename, primary_key, chunk_size):
    # Prepare chunk details
    chunk_number = 1
    chunk_data_size = 0

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)  # load whole JSON

    data_chunk = {"data": []}  # added this
    
    if isinstance(data, list):  # If data is a list, assign to records
        records = data
    else:  # If data is dictionary, assign to record_values
        records = data.values()

    for record in records:
        primary_key_value = get_nested_value(primary_key, record)
        record_dict = {primary_key_value: record}
        record_str = json.dumps(record_dict)
        record_size = len(record_str.encode('utf-8'))  # size in bytes

        if (record_size + chunk_data_size > chunk_size):
            chunk_file = file_for_chunk(filename, chunk_number)
            with open(chunk_file, 'w') as chunk:
                json.dump(data_chunk, chunk)

            data_chunk = {"data": []}  # Start a new chunk
            chunk_data_size = 0  # Reset the chunk data size
            chunk_number += 1  # Move on to the next chunk

        data_chunk["data"].append(record_dict)
        chunk_data_size += record_size

    # Write the last chunk if not empty
    if data_chunk["data"]:
        chunk_file = file_for_chunk(filename, chunk_number)
        with open(chunk_file, 'w') as chunk:
            json.dump(data_chunk, chunk)

def file_for_chunk(filename, chunk_number):
    # Get the file basename and remove the .json extension
    basename = os.path.splitext(os.path.basename(filename))[0]

    # Prepare the dataset and chunk directory
    dataset_dir = os.path.join('database', basename)
    os.makedirs(dataset_dir, exist_ok=True)  # Create directories recursively

    # Prepare the chunk filename
    chunk_file = os.path.join(dataset_dir, f"{chunk_number}.json")

    return chunk_file

def insert_data(filename, new_data, primary_key, chunk_size):
    basename = os.path.splitext(os.path.basename(filename))[0]
    dataset_dir = os.path.join('database', basename)

    new_data_json = json.dumps(new_data)
    new_data_size = len(new_data_json.encode('utf-8'))

    chunk_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.json')], key=lambda x : int(os.path.splitext(x)[0]))

    if chunk_files:
        last_chunk_file = os.path.join(dataset_dir, chunk_files[-1])
        
        with open(last_chunk_file, 'r') as f:
            chunk = json.load(f)
            
        chunk['data'].append(new_data)  # try to add data
            
        with open(last_chunk_file + 'temp', 'w') as f:  # create temporary new file 
            json.dump(chunk, f, indent=4)  # if chunk size is not exceeded, dump to file
            
        last_chunk_file_size = os.path.getsize(last_chunk_file + 'temp')  # get the real size of the chunk file
    
        if last_chunk_file_size > chunk_size:  # if chunk_size is exceeded after appending new_data
            chunk['data'].remove(new_data)  # remove the new data

            os.remove(last_chunk_file + 'temp')  # remove the temporary file

            with open(last_chunk_file + 'temp', 'w') as f:
                json.dump(chunk, f, indent=4)  # rewrite the file without new_data

            create_new_chunk(dataset_dir, new_data, chunk_files)  # create new chunk
            os.remove(last_chunk_file + 'temp')  # remove the temporary file after it's not needed
        else:
            os.remove(last_chunk_file)  # remove old file
            os.rename(last_chunk_file + 'temp', last_chunk_file)  # rename new file to old file

    else:
        create_new_chunk(dataset_dir, new_data)


def create_new_chunk(dataset_dir, new_data, chunk_files=[]):
    new_chunk_number = (int(os.path.splitext(chunk_files[-1])[0]) + 1) if chunk_files else 1
    new_chunk = {"data": [new_data]}
    new_chunk_file = os.path.join(dataset_dir, f'{new_chunk_number}.json')

    with open(new_chunk_file, 'w') as f:
        json.dump(new_chunk, f, indent=4)

def find_data(dataset, primary_key_value, primary_key):
    dataset_dir = os.path.join('database', dataset)
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.json')], key=lambda x : int(os.path.splitext(x)[0]))
  
    for file in files:
        with open(os.path.join(dataset_dir, file), 'r', encoding='utf-8') as f:
            data_list = json.load(f)['data']  # get the data inside the 'data' key

        for data in data_list:
            for obj in data.values():  # iterate over objects stored under each key in data
                if isinstance(primary_key, list): 
                    # For if the key is nested
                    nested_obj = obj
                    for subkey in primary_key:
                        nested_obj = nested_obj.get(subkey, {})
                    if primary_key_value == nested_obj:
                        return obj
                elif primary_key_value == obj.get(primary_key):
                    return obj
    return None

def delete_data(dataset, query, chunk_size):
    dataset_dir = os.path.join('database', dataset)
    chunk_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.json')], key=lambda x: int(os.path.splitext(x)[0]))
    
    deleted_data = []  # to store deleted data
    temp_files = []  # to store paths to temporary files
    current_size = 0  # Initialize current size to track the size of the deleted data

    def write_to_temp_file(data):
        fd, temp_filename = tempfile.mkstemp(suffix='.json', dir=dataset_dir)
        with open(temp_filename, 'w', encoding='utf-8') as temp_file:
            json.dump(data, temp_file)
        os.close(fd)
        return temp_filename

    def append_or_write_to_temp(data, obj, current_size):
        obj_json = json.dumps(obj)
        obj_size = len(obj_json.encode('utf-8'))

        if current_size + obj_size > chunk_size:
            temp_files.append(write_to_temp_file(data))
            return [obj], obj_size  # Start new list with the new object, reset size
        else:
            data.append(obj)
            return data, current_size + obj_size  # Append object and update current size

    for file in chunk_files:
        file_dir = os.path.join(dataset_dir, file)
        with open(file_dir, 'r', encoding='utf-8') as f:
            data_list = json.load(f)['data']

        updated_data_list = []
        for data in data_list:
            for _, obj in data.items():
                if not check_query_conditions(obj, query):
                    updated_data_list.append(data)
                else:
                    deleted_data, current_size = append_or_write_to_temp(deleted_data, obj, current_size)

        with open(file_dir, 'w', encoding='utf-8') as f:
            json.dump({"data": updated_data_list}, f)

    if deleted_data:
        temp_files.append(write_to_temp_file(deleted_data))

    # Print and clean up temporary files
    for temp_file in temp_files:
        with open(temp_file, 'r', encoding='utf-8') as f:
            temp_data = json.load(f)
            print("Deleted Data:", temp_data)

        os.remove(temp_file)


def check_query_conditions(obj, query):
    for condition in query:
        operator = condition['operator']
        key = condition['key']
        value = str(condition['value'])
        keys = key.split(',')

        if len(keys) > 1:
            nested_obj = obj
            for subkey in keys:
                nested_obj = nested_obj.get(subkey.strip(), {})
            if nested_obj == {}:
                return False
            obj_value = str(nested_obj)
        else:
            obj_value = str(obj.get(key, ''))

        if operator in ['contains', 'startswith', 'endswith', '=']:
            if operator == 'contains':
                if value not in obj_value:
                    return False
            elif operator == 'startswith':
                if not obj_value.startswith(value):
                    return False
            elif operator == 'endswith':
                if not obj_value.endswith(value):
                    return False
            elif operator == '=' and obj_value != value:
                return False
        else:
            if obj_value.isdigit():
                obj_value = int(obj_value)
                value = int(value)

                if operator == '>':
                    if obj_value <= value:
                        return False
                elif operator == '<':
                    if obj_value >= value:
                        return False
                elif operator == '>=':
                    if obj_value < value:
                        return False
                elif operator == '<=':
                    if obj_value > value:
                        return False
                elif operator == '!=':
                    if obj_value == value:
                        return False
            else:
                return False  # If non-digit, do not select.
    return True

def select_data(dataset, query, group_by_attr=None, attributes=None, limit=None, aggregate_instructions=None, CHUNK_SIZE=1024 * 1024):
    dataset_dir = os.path.join('database', dataset)
    chunk_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.json')], key=lambda x : int(os.path.splitext(x)[0]))

    temp_files = []
    total_records = 0

    for file in chunk_files:
        with open(os.path.join(dataset_dir, file), 'r', encoding='utf-8') as f:
            data = json.load(f)

        selected_chunk_data = []
        for record in data['data']:
            for _, obj in record.items():
                if check_query_conditions(obj, query):
                    if attributes is None or (group_by_attr and aggregate_instructions):
                        selected_obj = obj
                    else:
                        selected_obj = project_attributes(obj, attributes)

                    selected_chunk_data.append(selected_obj)
                    total_records += 1
                    if limit is not None and total_records >= limit:
                        break

            if limit is not None and total_records >= limit:
                break

        if selected_chunk_data:
            # Save selected data to temporary file.
            temp_filename = write_to_temp_file(selected_chunk_data, directory='database/temp')
            temp_files.append(temp_filename)

        if limit is not None and total_records >= limit:
            break

    # Proceed with grouping and aggregation using temp files
    if group_by_attr:
        group_files = group_data(temp_files, group_by_attr)
        if aggregate_instructions:
            return aggregate_grouped_data(group_files, aggregate_instructions)
        else:
            return print_data_from_temp_files(group_files)  # Function to print data from group-specific temp files

    return print_data_from_temp_files2(temp_files)  # Function to print data from temp files


def group_data(temp_files, group_by_attr):
    # Accumulate records for each group in memory first
    grouped_records = {}

    for temp_file in temp_files:
        with open(temp_file, 'r') as f:
            data = json.load(f)
        
        for record in data:
            group_key = get_nested_value(group_by_attr.split(','), record)
            if group_key not in grouped_records:
                grouped_records[group_key] = []

            grouped_records[group_key].append(record)

    # Write accumulated records for each group to separate temporary files
    group_temp_files = {}
    for group_key, records in grouped_records.items():
        temp_filename = write_to_temp_file(records, directory='database/temp')
        group_temp_files[group_key] = temp_filename

    return group_temp_files
    
def aggregate_grouped_data(group_files, aggregate_instructions):
    aggregated_results = {}
    for group_key, temp_file in group_files.items():
        with open(temp_file, 'r') as f:
            records = json.load(f)

        aggregated_results[group_key] = {}
        for instruction in aggregate_instructions:
            operation, attr = instruction.split(':')
            if operation == 'sum':
                aggregated_results[group_key][f'sum({attr})'] = sum(get_nested_value(attr.split(','), record) for record in records)
            elif operation == 'count':
                aggregated_results[group_key][f'count({attr})'] = len([record for record in records if get_nested_value(attr.split(','), record) is not None])
            elif operation == 'avg':            
                values = [get_nested_value(attr.split(','), record) for record in records]
                aggregated_results[group_key][f'avg({attr})'] = sum(values) / len(values)
            elif operation == 'max':
                values = [get_nested_value(attr.split(','), record) for record in records]
                aggregated_results[group_key][f'max({attr})'] = max(values)
            elif operation == 'min':
                values = [get_nested_value(attr.split(','), record) for record in records]
                aggregated_results[group_key][f'min({attr})'] = min(values)
    return aggregated_results  # This can be printed or written to a file as required


def project_data(dataset, query, attributes=None, CHUNK_SIZE=1024 * 1024):
    dataset_dir = os.path.join('database', dataset)
    chunk_files = sorted([f for f in os.listdir(dataset_dir) if os.path.splitext(f)[0].isdigit()], key=lambda x: int(os.path.splitext(x)[0]))

    temp_files = []

    for file in chunk_files:
        with open(os.path.join(dataset_dir, file), 'r', encoding='utf-8') as f:
            data_list = json.load(f)['data']

        selected_chunk_data = []
        for data in data_list:
            for _, obj in data.items():
                if check_query_conditions(obj, query):
                    projected_obj = project_attributes(obj, attributes)
                    selected_chunk_data.append(projected_obj)

        if selected_chunk_data:
            # Save projected data to temporary file.
            fd, temp_filename = tempfile.mkstemp()
            with open(temp_filename, 'w') as f:
                json.dump(selected_chunk_data, f)
            os.close(fd)

            temp_files.append(temp_filename)

    # Merge files.
    projected_data = []
    for temp_filename in temp_files:
        with open(temp_filename, 'r') as f:
            projected_data.extend(json.load(f))

    # Delete temporary files.
    for temp_filename in temp_files:
        if os.path.isfile(temp_filename):
            Path(temp_filename).unlink(missing_ok=True)

    return projected_data

def project_attributes(obj, attributes):
    projected_dict = {}
    for attr in attributes:
        nested_keys = attr.split(',')
        value = get_nested_value(nested_keys, obj)
        if len(nested_keys)>1:
            if nested_keys[0] not in projected_dict:
                projected_dict[nested_keys[0]] = {}
            projected_dict[nested_keys[0]][nested_keys[-1]] = value
        else:
            projected_dict[attr] = value
    return projected_dict

def aggregate_data(dataset, filter_query, target_key, aggregate_function, group_by_key=None, order_key=None, CHUNK_SIZE=1024 * 1024): 
    dataset_dir = os.path.join('database', dataset)
    chunk_files = sorted([f for f in os.listdir(dataset_dir) if os.path.splitext(f)[0].isdigit()], key=lambda x : int(os.path.splitext(x)[0]))
    aggregated_data = {}
    
    for file in chunk_files:
        with open(os.path.join(dataset_dir, file), 'r', encoding='utf-8') as f:
            data_list = json.load(f)['data']

        selected_chunk_data = []
        for data in data_list:
            for _, obj in data.items():
                if check_query_conditions(obj, filter_query):
                    selected_chunk_data.append(obj)

        if selected_chunk_data:
            for data in selected_chunk_data:
                if group_by_key:
                    group_by_value = get_nested_value(group_by_key.split(','), data) if ',' in group_by_key else data.get(group_by_key, '_nogroup')
                    if group_by_value not in aggregated_data:
                        aggregated_data[group_by_value] = []
                    target_value = get_nested_value(target_key.split(','), data) if ',' in target_key else data.get(target_key, 0) 
                    if isinstance(target_value, str) and target_value.strip().isdigit():
                        target_value = int(target_value)
                    aggregated_data[group_by_value].append(target_value)
                else:
                    target_value = get_nested_value(target_key.split(','), data) if ',' in target_key else data.get(target_key, 0) 
                    if target_value.strip().isdigit():  # Convert to int if it's numeric
                       target_value = int(target_value)
                    aggregated_data['_nogroup'].append(target_value)

    if aggregate_function == 'sum':
        for group in aggregated_data:
            aggregated_data[group] = sum(aggregated_data[group])
    elif aggregate_function == 'count':
        for group in aggregated_data:
            aggregated_data[group] = len(aggregated_data[group])
    elif aggregate_function == 'avg':
        for group in aggregated_data:
            aggregated_data[group] = sum(aggregated_data[group])/len(aggregated_data[group])
    elif aggregate_function == 'none':
        pass

    if order_key:
        aggregated_data = dict(sorted(aggregated_data.items(), key = lambda x:x[1]))
        
    return aggregated_data

def select2_data(dataset, query, attributes=None, limit=None, order_by_key=None, sort_order='asc', CHUNK_SIZE=1024 * 1024):
    dataset_dir = os.path.join('database', dataset)
    chunk_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.json')], key=lambda x: int(os.path.splitext(x)[0]))

    # Determine the size of one record
    with open(os.path.join(dataset_dir, chunk_files[0]), 'r', encoding='utf-8') as f:
        data = json.load(f)['data'][0]
        record_size = sys.getsizeof(json.dumps(data))

    max_files_in_memory = CHUNK_SIZE // record_size
    temp_files = []

    for i in range(0, len(chunk_files), max_files_in_memory):
        batch_files = chunk_files[i:i + max_files_in_memory]
        sorted_data = process_batch(batch_files, dataset_dir, query, attributes, order_by_key, sort_order)
        temp_filename = write_to_temp_file(sorted_data)
        temp_files.append(temp_filename)

    # Merge batches in multiple iterations until one file remains
    while len(temp_files) > 1:
        temp_files = merge_batches(temp_files, order_by_key, sort_order, CHUNK_SIZE)

    # Read the final sorted data
    with open(temp_files[0], 'r') as f:
        final_sorted_data = json.load(f)

    if limit:
        final_sorted_data = final_sorted_data[:limit]

    # Cleanup temp file
    if os.path.exists(temp_files[0]):
        os.remove(temp_files[0])

    return final_sorted_data

def merge_batches(temp_files, order_by_key, sort_order, CHUNK_SIZE):
    max_files_in_memory = CHUNK_SIZE // sys.getsizeof(json.dumps({}))  # Estimate max files based on an empty json object size
    new_temp_files = []

    for i in range(0, len(temp_files), max_files_in_memory):
        batch_files = temp_files[i:i + max_files_in_memory]
        merged_data = merge_sorted_temp_files(batch_files, order_by_key, sort_order)
        temp_filename = write_to_temp_file(merged_data)
        new_temp_files.append(temp_filename)

        # Cleanup old temp files
        for temp_file in batch_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    return new_temp_files


def process_batch(batch_files, dataset_dir, query, attributes, order_by_key, sort_order):
    batch_data = []
    for file in batch_files:
        with open(os.path.join(dataset_dir, file), 'r', encoding='utf-8') as f:
            data = json.load(f)['data']
            for record in data:
                for _, obj in record.items():
                    if check_query_conditions(obj, query):
                        batch_data.append(project_attributes(obj, attributes) if attributes else obj)

    if order_by_key:
        order_keys = order_by_key.split(',')
        batch_data.sort(key=lambda x: nested_sort_key_function(get_nested_value2(order_keys, x)), reverse=(sort_order == 'desc'))
    
    return batch_data

def get_nested_value2(nested_keys, data_dict):
    for key in nested_keys:
        if isinstance(data_dict, dict):
            data_dict = data_dict.get(key)
        else:
            return None
    return data_dict

def nested_sort_key_function(value):
    # If value is None, treat it as less than any other value
    return float('-inf') if value is None else value

def merge_sorted_temp_files(temp_files, order_by_key, sort_order):
    file_handles = [open(file, 'r') for file in temp_files]
    heapq_data = []

    # Check if order_by_key is provided
    if order_by_key is not None:
        order_keys = order_by_key.split(',')

    # Initialize the heap with the first element from each file
    for idx, f in enumerate(file_handles):
        data_iter = iter(json.load(f))
        try:
            first_element = next(data_iter)
            if order_by_key is not None:
                key_value = get_nested_value2(order_keys, first_element)
                heapq_key = -key_value if sort_order == 'desc' and key_value is not None else key_value
            else:
                heapq_key = 0  # Default value if no sorting is required
            heapq.heappush(heapq_data, (heapq_key, idx, first_element, data_iter))
        except StopIteration:
            continue

    sorted_data = []
    while heapq_data:
        _, file_idx, smallest, data_iter = heapq.heappop(heapq_data)
        sorted_data.append(smallest)

        try:
            next_element = next(data_iter)
            if order_by_key is not None:
                next_key_value = get_nested_value2(order_keys, next_element)
                heapq_key = -next_key_value if sort_order == 'desc' and next_key_value is not None else next_key_value
            else:
                heapq_key = 0  # Default value if no sorting is required
            heapq.heappush(heapq_data, (heapq_key, file_idx, next_element, data_iter))
        except StopIteration:
            continue

    # Clean up file handles
    for f in file_handles:
        f.close()

    return sorted_data

def write_to_temp_file(data, directory='database/temp'):
    # Ensure the specified directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Create a temporary file within the specified directory
    fd, temp_filename = tempfile.mkstemp(dir=directory, suffix='.json')
    with open(temp_filename, 'w') as f:
        json.dump(data, f)
    os.close(fd)
    return temp_filename

def sort_chunk(chunk, order_by_key, sort_order):
    order_keys = order_by_key.split(',')
    return sorted(chunk, key=lambda x: sort_key_function(get_nested_value(order_keys, x)), reverse=(sort_order == 'desc'))

def write_chunk_to_temp_file(chunk):
    fd, temp_filename = tempfile.mkstemp()
    with open(temp_filename, 'w') as f:
        json.dump(chunk, f)
    os.close(fd)
    return temp_filename

def merge_sorted_chunks(chunk_files, chunk_size, order_by_key, sort_order):
    merged_data = []
    chunk_iterators = [None] * len(chunk_files)
    current_elements = [None] * len(chunk_files)
    file_handles = [open(file, 'r') for file in chunk_files]

    # Initialize iterators for each chunk file
    for i, handle in enumerate(file_handles):
        chunk_iterators[i] = iter(json.load(handle))
        current_elements[i] = next(chunk_iterators[i], None)

    # Merge process
    while any(current_elements):
        # Identify the next element to merge
        current_min, current_index = None, None
        for i, element in enumerate(current_elements):
            if element is not None:
                current_val = sort_key_function(get_nested_value(order_by_key.split(','), element))
                if current_min is None or (sort_order == 'asc' and current_val < current_min) or (sort_order == 'desc' and current_val > current_min):
                    current_min = current_val
                    current_index = i

        # Add the identified element to merged data and load the next element from the same chunk
        if current_index is not None:
            merged_data.append(current_elements[current_index])
            current_elements[current_index] = next(chunk_iterators[current_index], None)

            # Check if memory limit is exceeded, if so, write to disk and clear memory
            if sys.getsizeof(merged_data) >= chunk_size:
                write_chunk_to_temp_file(merged_data)
                merged_data = []

    # Close file handles
    for handle in file_handles:
        handle.close()

    # Remove temporary chunk files
    for file in chunk_files:
        os.remove(file)

    return merged_data

def sort_key_function(value):
    try:
        return int(value)
    except ValueError:
        return value
    
def update_nested_key(obj, nested_keys, new_value):
    for key in nested_keys[:-1]:  # Go to the last key
        obj = obj.setdefault(key, {})
    obj[nested_keys[-1]] = new_value

def remove_nested_key(obj, nested_keys):
    for key in nested_keys[:-1]:
        obj = obj.get(key, {})
        if not obj:  # Exit if any key in the path does not exist
            return
    obj.pop(nested_keys[-1], None)

def update_data(dataset, query, updates, removes, chunk_size):
    dataset_dir = os.path.join('database', dataset)
    chunk_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.json')], key=lambda x: int(os.path.splitext(x)[0]))

    for file in chunk_files:
        file_dir = os.path.join(dataset_dir, file)
        with open(file_dir, 'r', encoding='utf-8') as f:
            data_list = json.load(f)['data']

        updated_chunk = []
        for data in data_list:
            for _, obj in data.items():
                if check_query_conditions(obj, query):
                    for update in updates:
                        for key, value in update.items():
                            nested_keys = key.split(',')
                            # Check and convert the data type
                            value = convert_value(obj, nested_keys, value)
                            update_nested_key(obj, nested_keys, value)
                    for remove in removes:
                        nested_keys = remove.split(',')
                        remove_nested_key(obj, nested_keys)
                updated_chunk.append(data)

        with open(file_dir, 'w', encoding='utf-8') as f:
            json.dump({"data": updated_chunk}, f)

def convert_value(obj, nested_keys, value):
    current_value = get_nested_value(nested_keys, obj)
    if isinstance(current_value, int):
        return int(value)
    elif isinstance(current_value, float):
        return float(value)
    # Add other data types as necessary
    return value

def join_datasets(dataset1, dataset2, join_key1, join_key2, chunk_size):
    dataset_dir1 = os.path.join('database', dataset1)
    dataset_dir2 = os.path.join('database', dataset2)

    temp_files = []  # To store temporary files of joined chunks

    for file1 in sorted(os.listdir(dataset_dir1)):
        with open(os.path.join(dataset_dir1, file1), 'r') as f1:
            data_chunk1 = json.load(f1)['data']

        for file2 in sorted(os.listdir(dataset_dir2)):
            with open(os.path.join(dataset_dir2, file2), 'r') as f2:
                data_chunk2 = json.load(f2)['data']

            joined_chunk, current_size = [], 0
            for record1 in data_chunk1:
                for key1, value1 in record1.items():
                    for record2 in data_chunk2:
                        for key2, value2 in record2.items():
                            if get_nested_value(join_key1, value1) == get_nested_value(join_key2, value2):
                                joined_record = {**value1, **value2}
                                joined_chunk.append(joined_record)
                                current_size += len(json.dumps(joined_record))

                                if current_size >= chunk_size:
                                    temp_file = write_temporary_file(joined_chunk)
                                    temp_files.append(temp_file)
                                    joined_chunk, current_size = [], 0

            if joined_chunk:  # Write remaining data to a temporary file
                temp_file = write_temporary_file(joined_chunk, temp_dir='database/temp')
                temp_files.append(temp_file)

    print_data_from_temp_files2(temp_files)

    # Cleanup temporary files
    temp_dir = 'database/temp'
    shutil.rmtree(temp_dir, ignore_errors=True)

def write_temporary_file(data, temp_dir='database/temp'):
    os.makedirs(temp_dir, exist_ok=True)  # Create the temp directory if it doesn't exist
    fd, temp_filename = tempfile.mkstemp(dir=temp_dir)
    with open(temp_filename, 'w') as temp_file:
        json.dump(data, temp_file)
    os.close(fd)
    return temp_filename

def print_data_from_temp_files(group_files):
    for group_key, temp_file in group_files.items():
        with open(temp_file, 'r') as file:
            data = json.load(file)
            print(f"Data for group {group_key}:")
            for record in data:
                print(record)
            print()  # New line for separation between group data

def print_data_from_temp_files2(temp_files):
    for temp_file in temp_files:
        with open(temp_file, 'r') as file:
            data = json.load(file)
            # print("Data from file:", temp_file)
            for record in data:
                print(record)
            print()  # New line for separation between files

def parse_insert_command(command):
    pattern = r'insert "(.+?)" into db'
    match = re.match(pattern, command, re.IGNORECASE)
    return match.group(1) if match else None

def parse_create_db_command(command):
    pattern = r"create database named (\w+) with data to path as (\S+) and chunk size of (\d+) kb primary key of (.+)"
    match = re.match(pattern, command, re.IGNORECASE)
    return match.groups() if match else (None, None, None, None)

def parse_switch_db_command(command):
    match = re.match(r"switch to db (\w+)", command, re.IGNORECASE)
    return match.group(1) if match else None

def parse_search_command(command):
    pattern = r"search data with primary id \"?([^\"]+)\"?"
    match = re.match(pattern, command, re.IGNORECASE)
    return match.group(1) if match else None

def parse_delete_command(command):
    pattern = r"delete data where (.+)"
    match = re.match(pattern, command, re.IGNORECASE)
    if match:
        conditions_str = match.group(1)
        conditions = conditions_str.split(" and ")
        query = []
        for condition in conditions:
            parts = condition.split(" ")
            if len(parts) >= 3:
                key = parts[0]
                operator = parts[1]
                value = parts[2]
                if operator == "greater-than":
                    operator = ">"
                elif operator == "less-than":
                    operator = "<"
                elif operator == "equals-to":
                    operator = "="
                # Add more operators as needed
                query.append({'operator': operator, 'key': key, 'value': value})
        return query
    return None

def parse_update_command(command):
    pattern = r"update data where (.+?)(?: & set (.*?))?(?: & remove (.*?))?$"
    match = re.match(pattern, command, re.IGNORECASE)
    if match:
        query_str, updates_str, removes_str = match.groups()

        query = parse_query_conditions(query_str) if query_str else []
        updates = parse_updates(updates_str) if updates_str else []
        removes = [remove.strip() for remove in removes_str.split('and')] if removes_str else []

        return query, updates, removes
    return None, None, None

def parse_updates(updates_str):
    updates_list = updates_str.split(' and ')
    updates = {}
    for update in updates_list:
        key, value = update.split(' = ')
        updates[key.strip()] = value.strip().strip("'\"")  # Strips quotes if present
    return [updates]

def parse_query_conditions(conditions_str):
    conditions = conditions_str.split(" and ")
    query = []
    for condition in conditions:
        parts = condition.split(" ")
        if len(parts) >= 3:
            key, operator, value = parts[0], parts[1], parts[2]
            if operator == "greater-than":
                operator = ">"
            elif operator == "less-than":
                operator = "<"
            elif operator == "equals-to":
                operator = "="
            # Add more operators as needed
            query.append({'operator': operator, 'key': key, 'value': value})
    return query

def parse_join_command(command):
    pattern = r"join (\w+) having attribute (.+?) with (\w+) having attribute (.+)"
    match = re.match(pattern, command, re.IGNORECASE)
    if match:
        dataset1, join_key1, dataset2, join_key2 = match.groups()
        join_key1 = [key.strip() for key in join_key1.split(',')]
        join_key2 = [key.strip() for key in join_key2.split(',')]
        return dataset1, join_key1, dataset2, join_key2
    return None, None, None, None

def parse_select_command(command):
    pattern = r"select data with condition (.*?)(?: & selective attributes of (.*?))?(?: & apply limit of (\d+))?(?: & group by (.*?))?(?: & apply aggregate functions of (.*?))?$"
    match = re.match(pattern, command, re.IGNORECASE)

    if match:
        conditions_str, attributes_str, limit_str, group_by_str, aggregate_str = match.groups()
        conditions = parse_query_conditions(conditions_str) if conditions_str else []
        attributes = [attr.strip() for attr in attributes_str.split('and')] if attributes_str else None
        limit = int(limit_str) if limit_str else None
        group_by = group_by_str.split(',') if group_by_str else None
        
        # Adjusting the format of aggregate functions
        aggregates = format_aggregate_functions(aggregate_str) if aggregate_str else None
        return conditions, attributes, limit, group_by, aggregates

    return None, None, None, None, None

def format_aggregate_functions(aggregate_str):
    aggregate_functions = []
    for func in aggregate_str.split('and'):
        func = func.strip()
        if '(' in func and ')' in func:
            operation, attr = func.split('(')
            attr = attr.strip(')')
            aggregate_functions.append(f"{operation}:{attr}")
    return aggregate_functions

def parse_select2_command(command):
    pattern = r"Select (.+?)(?: & selective attributes of (.*?))?(?: & apply limit of (\d+))?(?: & order by (.*?))?$"
    match = re.match(pattern, command, re.IGNORECASE)

    if match:
        conditions_str, attributes_str, limit_str, order_by_str = match.groups()
        conditions = parse_query_conditions(conditions_str) if conditions_str else []

        # Ensure attributes are parsed correctly
        attributes = [attr.strip() for attr in attributes_str.split('and')] if attributes_str else None

        limit = int(limit_str) if limit_str else None
        order_by_key, sort_order = parse_order_by(order_by_str) if order_by_str else (None, 'asc')

        return conditions, attributes, limit, order_by_key, sort_order

    return None, None, None, None, None

def parse_order_by(order_by_str):
    parts = order_by_str.rsplit(' ', 1)
    if len(parts) == 2:
        return parts[0], parts[1].lower()
    return order_by_str, 'asc'

import json

def save_db_state(databases, file_path='db_state.json'):
    with open(file_path, 'w') as file:
        json.dump(databases, file)

def load_db_state(file_path='db_state.json'):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}  # Return an empty dictionary if the file doesn't exist

def parse_delete_db_command(command):
    pattern = r"delete database named (\w+)"
    match = re.match(pattern, command, re.IGNORECASE)
    return match.group(1) if match else None

def main():
    databases = load_db_state()
    current_db = None

    while True:
        prompt = f"MyDB{f' ({current_db})' if current_db else ''} > "
        user_input = input(prompt)

        if user_input.lower().startswith("create database named"):
            db_name, file_path, chunk_size, primary_key = parse_create_db_command(user_input)
            if db_name and file_path and chunk_size and primary_key:
                chunk_size = (int(chunk_size)-1) * 512  # Convert to bytes
                primary_key = [key.strip() for key in primary_key.split(',')] if ',' in primary_key else primary_key
                filename = os.path.join("dataset", file_path)
                segregate_data(filename, primary_key, chunk_size)
                databases[db_name] = {
                    "file_path": file_path,
                    "chunk_size": chunk_size,
                    "primary_key": primary_key
                }
                current_db = db_name  # Set the current database
                print(f"Database '{db_name}' created and data loaded.")
                save_db_state(databases)  # Save the state after creating a new database
            else:
                print("Invalid command format for creating databases.")

        elif user_input.lower().startswith("switch to db"):
            db_name = parse_switch_db_command(user_input)
            if db_name and db_name in databases:
                current_db = db_name
                print(f"Switched to database '{db_name}'.")
                save_db_state(databases)  # Save the state after switching the database
            elif db_name:
                print(f"Database '{db_name}' does not exist. Please create it first.")
            else:
                print("Invalid command format for switching databases.")
  
        elif user_input.lower().startswith("search data with primary id"):
            if not current_db:
                print("No database selected. Use 'switch to db' to select a database.")
            else:
                primary_key_value = parse_search_command(user_input)
                if primary_key_value:
                    try:
                        primary_key_value = int(primary_key_value)  # Assuming the primary key is an integer
                    except ValueError:
                        print("Invalid primary key value. It should be an integer.")
                        continue
                    
                    # Retrieve current database settings
                    db_settings = databases[current_db]
                    dataset = db_settings["file_path"].split('.')[0]  # Assuming file_path includes '.json'
                    primary_key = db_settings["primary_key"]
                    
                    # Call find_data
                    result = find_data(dataset, primary_key_value, primary_key)
                    if result:
                        print("Data found: ", result)
                    else:
                        print("No data found.")
                else:
                    print("Invalid command format for searching data.")

        elif user_input.lower().startswith("insert "):
            if not current_db:
                print("No database selected. Use 'switch to db' to select a database.")
            else:
                data_str = parse_insert_command(user_input)
                if data_str:
                    try:
                        new_data = eval(data_str)
                        if isinstance(new_data, dict):
                            db_settings = databases[current_db]
                            filename = db_settings["file_path"]
                            primary_key = db_settings["primary_key"]
                            chunk_size = db_settings["chunk_size"]

                            insert_data(filename, new_data, primary_key, chunk_size)
                            print("Data inserted successfully.")
                        else:
                            print("Invalid data format. Data should be a dictionary.")
                    except (SyntaxError, NameError):
                        print("Invalid data format.")
                else:
                    print("Invalid command format for inserting data.")
        
        elif user_input.lower().startswith("delete data where"):
            if not current_db:
                print("No database selected. Use 'switch to db' to select a database.")
            else:
                query = parse_delete_command(user_input)
                if query:
                    db_settings = databases[current_db]
                    dataset = db_settings["file_path"].split('.')[0]
                    chunk_size = db_settings["chunk_size"]

                    delete_data(dataset, query, chunk_size)
                    print("Data deleted.")
                else:
                    print("Invalid command format for deleting data.")

        elif user_input.lower().startswith("update data where"):
            if not current_db:
                print("No database selected. Use 'switch to db' to select a database.")
            else:
                query, updates, removes = parse_update_command(user_input)
                if query is not None and updates is not None and removes is not None:
                    db_settings = databases[current_db]
                    dataset = db_settings["file_path"].split('.')[0]
                    chunk_size = db_settings["chunk_size"]

                    update_data(dataset, query, updates, removes, chunk_size)
                    print("Data updated.")
                else:
                    print("Invalid command format for updating data.")
        
        elif user_input.lower().startswith("join"):
            if not current_db:
                print("No database selected. Use 'switch to db' to select a database.")
            else:
                dataset1, join_key1, dataset2, join_key2 = parse_join_command(user_input)
                if dataset1 and dataset2:
                    db_settings = databases[current_db]
                    chunk_size = db_settings["chunk_size"]

                    join_datasets(dataset1, dataset2, join_key1, join_key2, chunk_size)
                    print("Datasets joined successfully.")
                else:
                    print("Invalid command format for joining datasets.")

        elif user_input.lower().startswith("select data with condition"):
            if not current_db:
                print("No database selected. Use 'switch to db' to select a database.")
            else:
                conditions, attributes, limit, group_by, aggregates = parse_select_command(user_input)
                if conditions is not None:
                    db_settings = databases[current_db]
                    dataset = db_settings["file_path"].split('.')[0]
                    chunk_size = db_settings["chunk_size"]

                    # Convert group_by list back to a comma-separated string
                    group_by_str = ','.join(group_by) if group_by else None

                    selected_data = select_data(dataset, conditions, group_by_str, attributes, limit, aggregates, chunk_size)
                    if selected_data:
                        print("Selected data:", selected_data)
                else:
                    print("Invalid command format for selecting data.")
        
        elif user_input.lower().startswith("select "):
            if not current_db:
                print("No database selected. Use 'switch to db' to select a database.")
            else:
                conditions, attributes, limit, order_by_key, sort_order = parse_select2_command(user_input)
                if conditions is not None:
                    db_settings = databases[current_db]
                    dataset = db_settings["file_path"].split('.')[0]
                    chunk_size = db_settings["chunk_size"]

                    selected_data = select2_data(dataset, conditions, attributes, limit, order_by_key, sort_order, chunk_size)
                    if selected_data:
                        print("Selected data:", selected_data)
                else:
                    print("Invalid command format for selecting data.")

        elif user_input.lower() == "exit db":
            current_db = None
            save_db_state(databases)  # Save the state when exiting the current database
            print("Exited current database.")

        elif user_input.lower() == "exit":
            save_db_state(databases)  # Save the state before exiting the application
            break
        
        elif user_input.lower().startswith("delete database named"):
            db_name_to_delete = parse_delete_db_command(user_input)
            if db_name_to_delete and db_name_to_delete in databases:
                # Delete the database directory
                try:
                    db_dir = os.path.join('database', db_name_to_delete)
                    if os.path.exists(db_dir):
                        shutil.rmtree(db_dir)
                        print(f"Database directory '{db_name_to_delete}' deleted.")
                    else:
                        print(f"Database directory '{db_name_to_delete}' does not exist.")

                    # Remove database from the databases dictionary
                    del databases[db_name_to_delete]
                    if current_db == db_name_to_delete:
                        current_db = None
                    print(f"Database '{db_name_to_delete}' deleted from the system.")
                    save_db_state(databases)  # Save the state after deleting the database

                except Exception as e:
                    print(f"Error occurred while deleting the database: {e}")

            elif db_name_to_delete:
                print(f"Database '{db_name_to_delete}' does not exist.")
            else:
                print("Invalid command format for deleting databases.")

        else:
            print("Unknown command.")

if __name__ == "__main__":
    main()