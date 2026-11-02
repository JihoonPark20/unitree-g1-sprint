"""Camera controls shared by the Viser training and reference viewers."""

import threading

import numpy as np


class ViserCameraControls:
    def __init__(self, server, root, *, follow=True):
        self.server = server
        self._root = np.asarray(root, dtype=float).copy()
        self._lock = threading.RLock()
        with server.gui.add_folder("Camera"):
            self.follow = server.gui.add_checkbox("Follow robot", initial_value=follow)
            self.azimuth = server.gui.add_slider("Azimuth (deg)", min=-180, max=180, step=1, initial_value=-50)
            self.elevation = server.gui.add_slider("Elevation (deg)", min=-80, max=89, step=1, initial_value=20)
            self.distance = server.gui.add_slider("Distance (m)", min=0.5, max=15.0, step=0.1, initial_value=3.0)
            self.height = server.gui.add_slider("Target height offset (m)", min=-1.0, max=2.0, step=0.05, initial_value=0.0)
            for label, angles in (("Front", (0, 10)), ("Side", (-90, 10)), ("Rear", (180, 10)), ("Top", (-90, 89))):
                button = server.gui.add_button(label)

                @button.on_click
                def preset(_, angles=angles):
                    self.azimuth.value, self.elevation.value = angles
                    self.apply()

            focus = server.gui.add_button("Center on robot")
            reset = server.gui.add_button("Reset view")
            server.gui.add_markdown("Drag to orbit/pan; scroll to zoom. Follow preserves your adjusted view.")

        for control in (self.azimuth, self.elevation, self.distance, self.height):
            control.on_update(lambda _: self.apply())
        focus.on_click(lambda _: self.center())
        self.follow.on_update(lambda _: self.center() if self.follow.value else None)

        @reset.on_click
        def reset_view(_):
            self.azimuth.value = -50
            self.elevation.value = 20
            self.distance.value = 3.0
            self.height.value = 0.0
            self.apply()

        @server.on_client_connect
        def connect(client):
            with self._lock, server.atomic():
                self._apply_client(client)

    def _target(self):
        return self._root + np.array([0.0, 0.0, self.height.value])

    def _apply_client(self, client):
        azimuth, elevation = np.deg2rad([self.azimuth.value, self.elevation.value])
        offset = self.distance.value * np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ])
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = self._target() + offset
        client.camera.look_at = self._target()

    def apply(self):
        with self._lock, self.server.atomic():
            for client in self.server.get_clients().values():
                self._apply_client(client)

    def center(self):
        """Recenter without discarding a manually chosen orbit or zoom."""
        with self._lock, self.server.atomic():
            for client in self.server.get_clients().values():
                delta = self._target() - np.asarray(client.camera.look_at)
                client.camera.position = np.asarray(client.camera.position) + delta
                client.camera.look_at = self._target()

    def update_root(self, root):
        with self._lock, self.server.atomic():
            root = np.asarray(root, dtype=float)
            delta = root - self._root
            self._root = root.copy()
            if self.follow.value:
                for client in self.server.get_clients().values():
                    client.camera.position = np.asarray(client.camera.position) + delta
                    client.camera.look_at = np.asarray(client.camera.look_at) + delta
