from queue import Empty, SimpleQueue
from threading import Event

from PIL import Image, ImageDraw
from pystray import Icon, Menu, MenuItem


class WindowTray:
    def __init__(self, root, quit_app):
        self.root = root
        self.ready = Event()
        self.actions = SimpleQueue()

        image = Image.new("RGBA", (64, 64))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((2, 2, 61, 61), radius=14, fill="#39245d")
        draw.polygon([(25, 16), (25, 48), (48, 32)], fill="white")

        self.icon = Icon("club-coven", image, "Club Coven", Menu(
            MenuItem("ウィンドウを表示", lambda: self.actions.put(self.restore), default=True),
            MenuItem("終了", lambda: self.actions.put(quit_app)),
        ))
        self.icon.run_detached(setup=lambda icon: self.ready.set())
        self.root.after(50, self.poll)

    def restore(self):
        self.root.deiconify()
        self.root.lift()
        self.icon.visible = False

    def poll(self):
        # Tray callbacks run on the Windows message thread. Run Tk actions here.
        self.root.after(50, self.poll)
        if self.ready.is_set() and self.root.state() == "iconic":
            self.icon.visible = True
            self.root.withdraw()
        try:
            action = self.actions.get_nowait()
        except Empty:
            return
        action()

    def close(self):
        self.ready.wait()
        self.icon.stop()
