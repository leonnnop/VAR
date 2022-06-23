import json

def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)

def save_parsed_args_to_json(parsed_args, file_path, pretty=True):
    args_dict = vars(parsed_args)
    save_json(args_dict, file_path, save_pretty=pretty)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def merge_dicts(list_dicts):
    merged_dict = list_dicts[0].copy()
    for i in range(1, len(list_dicts)):
        merged_dict.update(list_dicts[i])
    return merged_dict


def merge_json_files(paths, merged_path):
    merged_dict = merge_dicts([load_json(e) for e in paths])
    save_json(merged_dict, merged_path)