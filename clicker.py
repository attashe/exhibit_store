import sys
from pynput.mouse import Controller, Button


def main():
    mouse = Controller()
    mouse.position = (int(sys.argv[1]), int(sys.argv[2]))

    mouse.click(Button.left, 1)


if __name__ == "__main__":
    main()