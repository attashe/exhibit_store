import sqlite3
conn = sqlite3.connect('example.db')


def database_init():
    c = conn.cursor()

    # Create table
    c.execute('''CREATE TABLE images
                (camera_id text, image_id text, image BLOB)''')

    # Insert a row of data
    # c.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")

    # Save (commit) the changes
    conn.commit()

    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    conn.close()


def convertToBinaryData(filename):
    #Convert digital data to binary format
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData


def insertBLOB(camera_id, image_id, photo):
    try:
        # sqliteConnection = sqlite3.connect('SQLite_Python.db')
        cursor = conn.cursor()
        print("Connected to SQLite")
        sqlite_insert_blob_query = """ INSERT INTO images
                                  (camera_id, image_id, image) VALUES (?, ?, ?)"""

        empPhoto = convertToBinaryData(photo)
        # resume = convertToBinaryData(resumeFile)
        # Convert data into tuple format
        data_tuple = (camera_id, image_id, empPhoto)
        cursor.execute(sqlite_insert_blob_query, data_tuple)
        conn.commit()
        print("Image and file inserted successfully as a BLOB into a table")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert blob data into sqlite table", error)
    finally:
        if (conn):
            conn.close()
            print("the sqlite connection is closed")


def save_image(camera, image_id, image_array):
    pass


if __name__ == "__main__":
    # database_init()

    insertBLOB('test_camera', 'test_image', 'test.jpg')
