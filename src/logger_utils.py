import csv

class CSVWriter:
    """
    for writing tabular data to a csv file
    """
    def __init__(self, file_name, column_names):
        """
        ---------
        Arguments
        ---------
        file_name : str
            full path of csv file
        column_names : list
            a list of columns names to be used to create the csv file
        """
        self.file_name = file_name
        self.column_names = column_names

        self.file_handle = open(self.file_name, "w", encoding="utf-8", newline="\n")
        self.writer = csv.writer(self.file_handle)

        self.write_header()
        print(f"{self.file_name} created successfully with header row")

    def write_header(self):
        """
        writes header into csv file
        """
        self.write_row(self.column_names)
        return

    def write_row(self, row):
        """
        writes a row into csv file
        """
        self.writer.writerow(row)
        return

    def close(self):
        """
        close the file
        """
        self.file_handle.close()
        return
