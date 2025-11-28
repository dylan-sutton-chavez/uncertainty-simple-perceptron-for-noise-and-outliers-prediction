from config.settings import JSON_FILES_DIR_NAME

from json import loads, dumps
from os import getcwd, path

class JsonFileStore:
    def __init__(self, json_file_name: str):
        """
        Initializes the JSON file path and ensures the json file exists.

        Args:
            json_file_name: str → File name.

        Output:
            None

        Time complexity → O(1)
        """
        self.json_file_path: str = path.join(getcwd(), JSON_FILES_DIR_NAME, f'{json_file_name}.jsonl')
        self._check_if_exists()

    def insert(self, to_insert: dict[str, any]):
        """
        Appends a new dictionary (JSON line) in the JSON file.

        Args:
            to_insert: dict[str, any] → Dictionary containing the data record to be inserted as JSON.

        Output:
            None

        Time complexity → O(1)
        """
        self._check_if_exists()
        with open(self.json_file_path, 'a', encoding='utf-8') as json_file:
            json_file.write(f'{dumps(to_insert)}\n')

    def line_by_line(self):
        """
        Reads the JSON file line by line, yielding dictionary objects.

        Args:
            None

        Output:
            dict[str, any] → A dictionary object parsed from the JSON file line.

        Time complexity → O(l)
        """
        self._check_if_exists()
        with open(self.json_file_path, 'r', encoding='utf-8') as json_file:
            for line in json_file:
                yield loads(line)
    
    def truncate(self):
        """
        Clears all existing content from the current json file.

        Args:
            None

        Output:
            None

        Time complexity → O(1)
        """
        self._check_if_exists()
        with open(self.json_file_path, 'w'):
            pass
    
    def _check_if_exists(self):
        """
        Confirms the JSON file exists, creating if it is missing.

        Args:
            None

        Output:
            None

        Time complexity → O(1)
        """
        with open(self.json_file_path, 'a', encoding='utf-8') as json_file:
            json_file.write('')

if __name__ == '__main__':
    """
    Code block that runs when the script is executed directly.

    Time complexity → O(l)

    Run command (as a package '-m' and without 'byte-compile' -B): 
        python -B -m src.data.persistence.json_store
    """
    json_db = JsonFileStore("gate_or")
    json_db.truncate()

    gate_or_examples: dict[str, list[float] | float] = [
        {"vector": [0, 0], "target": 0},
        {"vector": [0, 1], "target": 1},
        {"vector": [1, 0], "target": 1},
        {"vector": [1, 1], "target": 1}
    ]

    for example in gate_or_examples:
        json_db.insert(example)

    for line in json_db.line_by_line():
        target: int = line["target"]

        if target == 1:
            print(line)
            continue