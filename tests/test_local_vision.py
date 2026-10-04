import unittest
from modules.local_vision import server_command

class BackendTests(unittest.TestCase):
    def test_cpu_disables_language_and_vision_gpu_offload(self):
        command = server_command("server", "model", "projector", "cpu")
        self.assertIn("--no-mmproj-offload", command)
        self.assertEqual(command[command.index("--device") + 1], "none")
        self.assertEqual(command[command.index("-ngl") + 1], "0")

    def test_metal_and_cuda_use_discovered_devices(self):
        for backend, device in (("metal", "MTL0"), ("cuda", "CUDA0")):
            command = server_command("server", "model", "projector", backend, devices=f"  {device}: GPU")
            self.assertEqual(command[command.index("--device") + 1], device)

    def test_cuda_fails_explicitly_on_mac(self):
        with self.assertRaises(ValueError):
            server_command("server", "model", "projector", "cuda", devices="MTL0: M4")

    def test_thread_count_can_be_tuned_for_windows_cpu(self):
        command = server_command("server", "model", "projector", "cpu", threads=8)
        self.assertEqual(command[command.index("-t") + 1], "8")
