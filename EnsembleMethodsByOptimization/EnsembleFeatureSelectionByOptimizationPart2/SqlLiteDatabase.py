import sqlite3


class SqlLiteDatabase:

    #######################################################################
    ## The constructor of the SqlLiteDatabase class
    #  @param name Optionally, the name of the database to open.
    #######################################################################
    def __init__(self, name=None):

        self.conn = None
        self.cursor = None

        if name:
            self.open(name)

    #######################################################################
    ## Opens a new database connection.
    #  This function manually opens a new database connection.
    #  @param name Optionally, the name of the database to open.
    #######################################################################

    def open(self, name):
        try:
            self.conn = sqlite3.connect(name);
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            print("Error connecting to database!")

    #######################################################################
    ## Function to close a datbase connection.
    #######################################################################
    def close(self):

        if self.conn:
            self.conn.commit()
            self.cursor.close()
            self.conn.close()

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        self.close()

    #######################################################################
    ## Function to fetch/query data from a database.
    #  @param table The name of the database's table
    #  @param columns The string of columns, comma-separated, to fetch.
    #  @param limit Optionally, a limit of items to fetch.
    #######################################################################
    def get(self, table, columns, limit=None):

        query = "SELECT {0} from {1};".format(columns, table)
        self.cursor.execute(query)

        # fetch data
        rows = self.cursor.fetchall()

        return rows[len(rows) - limit if limit else 0:]

    #######################################################################
    ## Utilty function to get the ith row of data from a database.
    #  @param table The name of the database's table
    #  @param columns The columns which to query.
    #######################################################################
    def getLast(self, table, columns,i):

        return self.get(table, columns, limit=1)[i]

    #######################################################################
    ## Utility function that converts a dataset into CSV format.
    #  @param data The data, retrieved from the get() function.
    #  @param fname The file name to store the data in.
    #######################################################################
    @staticmethod
    def toCSV(data, fname="output.csv"):

        with open(fname, 'a') as file:
            file.write(",".join([str(j) for i in data for j in i]))

    #######################################################################
    ## Function to write data to the database.
    #  @param table The name of the database's table to write to.
    #  @param columns The columns to insert into, as a comma-separated string.
    #  @param data The new data to insert, as a comma-separated string.
    #######################################################################
    def write(self, table, columns, data):

        query = "INSERT INTO {0} ({1}) VALUES ({2});".format(table, columns, data)
        self.cursor.execute(query)

    #######################################################################
    ## Function to query any other SQL statement.
    #  This function is there in case you want to execute any other sql
    #  statement which is not write and get.
    #  @param sql A valid SQL statement in string format.
    #######################################################################
    def query(self, sql):
        self.cursor.execute(sql)

    #######################################################################
    ## Utility function that summarizes a dataset.
    #  This function takes a dataset, retrieved via the get() function, and
    #  returns only the maximum, minimum and average for each column.
    #  @param rows The retrieved data.
    #######################################################################
    @staticmethod
    def summary(rows):

        # split the rows into columns
        cols = [[r[c] for r in rows] for c in range(len(rows[0]))]

        # the time in terms of fractions of hours of how long ago
        # the sample was assumes the sampling period is 10 minutes
        t = lambda col: "{:.1f}".format((len(rows) - col) / 6.0)

        # return a tuple, consisting of tuples of the maximum,
        # the minimum and the average for each column and their
        # respective time (how long ago, in fractions of hours)
        # average has no time, of course
        ret = []

        for c in cols:
            hi = max(c)
            hi_t = t(c.index(hi))

            lo = min(c)
            lo_t = t(c.index(lo))

            avg = sum(c) / len(rows)

            ret.append(((hi, hi_t), (lo, lo_t), avg))

        return ret
