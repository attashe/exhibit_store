import os
import sys
import yaml
import email
import shelve
import sqlite3
import imaplib
import webbrowser

import configargparse

from pathlib import Path
from loguru import logger

from email.header import decode_header
from email.policy import default

from utils import email_utils



def init_db():
    pass


def email_connect(username, password, server="imap.mail.ru"):
    # create an IMAP4 class with SSL 
    imap = imaplib.IMAP4_SSL(server)
    # authenticate
    imap.login(username, password)

    return imap


def get_emails(imap, post_folder="INBOX/ToMyself"):
    status, messages = imap.select(post_folder)

    return status, messages


def insertPhoto(db_path, camera_id, image_id, photo):
    try:
        conn = sqlite3.connect(db_path)
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


def get_email_list(messages, target_folder='emails'):
    Path(target_folder).mkdir(exist_ok=True)
    count_messages = int(messages[0])
    for i in range(count_messages, 0, -1):
        # fetch the email message by ID
        res, msg = imap.fetch(str(i), "(RFC822)")
        for response in msg:
            if isinstance(response, tuple):
                # parse a bytes email into a message object
                msg = email.message_from_bytes(response[1])
                # decode the email subject
                subject = decode_header(msg["Subject"])[0][0]
                if isinstance(subject, bytes):
                    # if it's a bytes, decode to str
                    subject = subject.decode()
                # email sender
                from_ = msg.get("From")
                print("Subject:", subject)
                print("From:", from_)
                # if the email message is multipart
                if msg.is_multipart():
                    # iterate over email parts
                    for part in msg.walk():
                        # extract content type of email
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        try:
                            # get the email body
                            body = part.get_payload(decode=True).decode()
                        except:
                            pass
                        if content_type == "text/plain" and "attachment" not in content_disposition:
                            # print text/plain emails and skip attachments
                            print(body)
                        elif "attachment" in content_disposition:
                            # download attachment
                            filename = part.get_filename()
                            if filename:
                                subpath = os.path.join(target_folder, subject)
                                if not os.path.isdir(subpath):
                                    # make a folder for this email (named after the subject)
                                    os.mkdir(subpath)
                                filepath = os.path.join(subpath, filename)
                                # download attachment and save it
                                open(filepath, "wb").write(part.get_payload(decode=True))
                else:
                    # extract content type of email
                    content_type = msg.get_content_type()
                    # get the email body
                    body = msg.get_payload(decode=True).decode()
                    if content_type == "text/plain":
                        # print only text email parts
                        print(body)
                if content_type == "text/html":
                    # if it's HTML, create a new HTML file and open it in browser
                    if not os.path.isdir(subject):
                        # make a folder for this email (named after the subject)
                        os.mkdir(subject)
                    filename = f"{subject[:50]}.html"
                    filepath = os.path.join(subject, filename)
                    # write the file
                    open(filepath, "w").write(body)
                    # open in the default browser
                    webbrowser.open(filepath)
                print("="*100)


def load_email():
    pass


def load_emails(store, messages, target_folder='emails'):
    count_messages = int(messages[0])
    for i in range(count_messages, 0, -1):
        index = str(i)
        if index not in store:
            temp = []
        else:
            continue
        # fetch the email message by ID
        res, msg = imap.fetch(str(i), "(RFC822)")
        for response in msg:
            if isinstance(response, tuple):
                # parse a bytes email into a message object
                msg = email.message_from_bytes(response[1])
                # decode the email subject
                subject = decode_header(msg["Subject"])[0][0]
                if isinstance(subject, bytes):
                    # if it's a bytes, decode to str
                    subject = subject.decode()
                # email sender
                from_ = msg.get("From")
                print("Subject:", subject)
                print("From:", from_)

                data = get_email_payload(subject, msg)
                temp.append(data)
        store[index] = temp


imgs_path = Path('imgs')
imgs_path.mkdir(exist_ok=True)


def get_email_payload(subject, msg):
    """Парсит бинарные данные письма и извлекает прикрепленные файлы

    Args:
        subject (str): Заголовок письма
        msg (object): Содержимое письма в бинарном формате
    """
    # if the email message is multipart
    data = {
        'subject': subject,
        'body': [],
        'attach': [],
    }
    if msg.is_multipart():
        # iterate over email parts
        for part in msg.walk():
            # extract content type of email
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            body = None
            try:
                # get the email body
                body = part.get_payload(decode=True).decode()
            except:
                pass
            if content_type == "text/plain" and "attachment" not in content_disposition:
                # print text/plain emails and skip attachments
                print(body)
                data['body'].append(body)
            elif "attachment" in content_disposition:
                # download attachment
                filename = part.get_filename()

                if filename:
                    data['body'].append(body)
                    data['attach'].append(filename)
                    with open(imgs_path / filename, "wb") as f:
                        f.write(part.get_payload(decode=True))
    # Not needed in our case               
    else:
        # extract content type of email
        content_type = msg.get_content_type()
        # get the email body
        body = msg.get_payload(decode=True).decode()
        if content_type == "text/plain":
            # print only text email parts
            print(body)
        data['body'].append(body)
    if content_type == "text/html":
        # if it's HTML, create a new HTML file and open it in browser
        raise Exception("HTML format doesn't supporting")
    
    print(data)
    return data



def create_db(db_path):
    print('Creating database')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table
    c.execute('''CREATE TABLE images
                (camera_id text, image_id text, image BLOB)''')

    # Save (commit) the changes
    conn.commit()

    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    conn.close()


def test_email_connetc(username, password):
    # create an IMAP4 class with SSL 
    imap = imaplib.IMAP4_SSL("imap.mail.ru")
    # authenticate
    imap.login(username, password)

    status, messages = imap.select("INBOX/ToMyself")

    # total number of emails
    messages = int(messages[0])

    imap.close()
    imap.logout()


def load_db_emails(db_path, ):
    rows = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print("Connected to SQLite")
        sqlite_select_query = """ SELECT * FROM images """

        cursor.execute(sqlite_select_query)
        rows = cursor.fetchall()
        
        for row in rows:
            logger.info(row)
    
    except sqlite3.Error as error:
        logger.error("Failed to select data into sqlite table", error)
    finally:
        if (conn):
            conn.close()
            logger.debug("the sqlite connection is closed")

    return rows


def check_email(emails_list):
    pass


def add_email(email, db, images_path):
    pass


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='Get images from email and ')
    parser.add_argument('-c', '-cfg', type=str, is_config_file=True,
                        help='path to config file')

    parser.add_argument('--image_folder', type=str,
                        help='path to validation folder')
    parser.add_argument('--database', type=str,
                        help='path to validation folder')
    parser.add_argument('--cred_path', type=str, default='./credentials.yaml',
                        help='path to credentials file')

    # init config
    args = parser.parse_args()
    # get credentials
    with open(args.cred_path, 'r') as f:
        credentials = yaml.load(f, Loader=yaml.SafeLoader)

    print(args, credentials)
    # authorize

    # load email list
    imap = email_connect(credentials['login'], credentials['pass'])

    # connect to DB
    if not Path(args.database).exists():
        create_db(args.database)

    # get emails from db
    logger.info('Opening database')
    # db_emails = load_db_emails(args.database)

    with shelve.open('emails_store.shelve') as store:
        # check new emails
        _, messages = get_emails(imap)
        print(messages)
        
        # load new emails
        # get_email_list(messages)
        load_emails(store, messages)

    # load attached images
    # save to DB and image folder
    pass
