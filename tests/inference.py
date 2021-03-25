import os
import sys

import configargparse


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
        credentials = yaml.load(f)

    print(args)
    print(credentials)
    # authorize
    # connect to DB
    if Path(args.database).exists():
        print('Creating database')
        conn = sqlite3.connect(args.database)
        c = conn.cursor()

        # Create table
        c.execute('''CREATE TABLE images
                    (camera_id text, image_id text, image BLOB)''')

        # Save (commit) the changes
        conn.commit()

        # We can also close the connection if we are done with it.
        # Just be sure any changes have been committed or they will be lost.
        conn.close()
    print('Opening database')

    
    
    # load email list
    # check new emails
    # load new emails
    # load attached images
    # save to DB and image folder
    pass
