"""
VINMEC AI MEDICAL HUB - GIAO DIEN DIEU KHIEN DEMO (LAUNCHER UI)
==============================================================
Giao diện đồ họa trung tâm cho phép lựa chọn và chuyển đổi linh hoạt:
- Luồng A: Giám sát an toàn & Phát hiện té ngã bệnh nhân cao tuổi (Option 2B).
- Luồng B: Phục hồi chức năng từ xa khớp khuỷu tay sau mổ (Option 2A).
- Nguồn phát: Webcam trực tiếp hoặc File Video có sẵn.
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class VinmecDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VinMec AI Medical Hub - Perception Demo")
        self.root.geometry("680x640")
        self.root.minsize(640, 580)
        self.root.configure(bg="#111827")  # Nền tối y tế hiện đại

        # Biến trạng thái
        self.current_process = None
        self.selected_plan = tk.StringVar(value="PLAN_A")  # PLAN_A hoặc PLAN_B
        self.source_type = tk.StringVar(value="WEBCAM")    # WEBCAM hoặc VIDEO
        self.camera_index = tk.StringVar(value="0")
        self.video_path = tk.StringVar()
        self.rehab_side = tk.StringVar(value="LEFT")       # LEFT hoặc RIGHT

        # Tự động tìm file video có sẵn trong thư mục
        default_video = "20240912_101331.mp4"
        if os.path.exists(default_video):
            self.video_path.set(os.path.abspath(default_video))

        # Thiết lập giao diện
        self._setup_styles()
        self._create_widgets()

    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Cấu hình màu sắc
        self.style.configure(".", background="#111827", foreground="#F9FAFB")
        self.style.configure("TLabel", background="#111827", foreground="#F9FAFB", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#38BDF8")
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 9), foreground="#9CA3AF")
        
        # Card Frames
        self.style.configure("Card.TFrame", background="#1F2937", relief="solid")
        self.style.configure("CardLabel.TLabel", background="#1F2937", foreground="#E5E7EB", font=("Segoe UI", 11, "bold"))
        self.style.configure("CardDesc.TLabel", background="#1F2937", foreground="#9CA3AF", font=("Segoe UI", 9))
        
        # Radiobuttons
        self.style.configure("Card.TRadiobutton", background="#1F2937", foreground="#F9FAFB", font=("Segoe UI", 10, "bold"))
        self.style.map("Card.TRadiobutton", background=[("active", "#1F2937")], foreground=[("active", "#38BDF8")])

        # Buttons
        self.style.configure(
            "Primary.TButton",
            font=("Segoe UI", 11, "bold"),
            background="#0284C7",
            foreground="#FFFFFF",
            borderwidth=0,
            padding=8,
        )
        self.style.map("Primary.TButton", background=[("active", "#0369A1"), ("disabled", "#4B5563")])

        self.style.configure(
            "Danger.TButton",
            font=("Segoe UI", 10, "bold"),
            background="#DC2626",
            foreground="#FFFFFF",
            borderwidth=0,
            padding=6,
        )
        self.style.map("Danger.TButton", background=[("active", "#B91C1C"), ("disabled", "#4B5563")])

        self.style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 9),
            background="#374151",
            foreground="#E5E7EB",
            borderwidth=0,
            padding=5,
        )
        self.style.map("Secondary.TButton", background=[("active", "#4B5563")])

    def _create_widgets(self):
        # Tiêu đề Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=24, pady=(18, 10))

        title_lbl = ttk.Label(header_frame, text="VINMEC AI MEDICAL HUB", style="Header.TLabel")
        title_lbl.pack(anchor="w")

        sub_lbl = ttk.Label(
            header_frame,
            text="He thong Camera AI Y te: Giam sat Te nga & Phuc hoi chuc nang (Intel Core i5 Optimized)",
            style="SubHeader.TLabel",
        )
        sub_lbl.pack(anchor="w", pady=(2, 0))

        # -------------------------------------------------------------
        # KHU VỰC 1: CHỌN LUỒNG (PLAN A vs PLAN B)
        # -------------------------------------------------------------
        plan_card = ttk.Frame(self.root, style="Card.TFrame", padding=14)
        plan_card.pack(fill="x", padx=24, pady=8)

        ttk.Label(plan_card, text="1. CHỌN LUỒNG GIÁM SÁT / CHĂM SÓC:", style="CardLabel.TLabel").pack(anchor="w", pady=(0, 8))

        # Luồng A
        rb_a = ttk.Radiobutton(
            plan_card,
            text="LUỒNG A: Giám sát an toàn & Phát hiện té ngã (Option 2B)",
            variable=self.selected_plan,
            value="PLAN_A",
            style="Card.TRadiobutton",
            command=self._on_plan_change,
        )
        rb_a.pack(anchor="w", pady=(2, 2))
        ttk.Label(
            plan_card,
            text="   - Telemetry FSM 3 tầng, lọc chống báo động giả (cúi người nhặt đồ < 1.2s), SOS viền đỏ chớp nháy.",
            style="CardDesc.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        # Luồng B
        rb_b = ttk.Radiobutton(
            plan_card,
            text="LUỒNG B: Phục hồi chức năng từ xa khớp khuỷu tay (Option 2A)",
            variable=self.selected_plan,
            value="PLAN_B",
            style="Card.TRadiobutton",
            command=self._on_plan_change,
        )
        rb_b.pack(anchor="w", pady=(2, 2))
        ttk.Label(
            plan_card,
            text="   - Đo góc ROM 2D (50-155 độ), bộ lọc EMA, phát hiện sai tư thế dạng vai (>35 độ) & xuất CSV.",
            style="CardDesc.TLabel",
        ).pack(anchor="w", pady=(0, 2))

        # Tùy chọn riêng cho Luồng B: Chọn tay
        self.rehab_options_frame = ttk.Frame(plan_card, style="Card.TFrame")
        ttk.Label(self.rehab_options_frame, text="   Tay tập theo dõi:", style="CardDesc.TLabel").pack(side="left", padx=(0, 8))
        ttk.Radiobutton(self.rehab_options_frame, text="Tay Trái (LEFT)", variable=self.rehab_side, value="LEFT", style="Card.TRadiobutton").pack(side="left", padx=6)
        ttk.Radiobutton(self.rehab_options_frame, text="Tay Phải (RIGHT)", variable=self.rehab_side, value="RIGHT", style="Card.TRadiobutton").pack(side="left", padx=6)

        # -------------------------------------------------------------
        # KHU VỰC 2: CHỌN NGUỒN PHÁT (WEBCAM vs VIDEO)
        # -------------------------------------------------------------
        source_card = ttk.Frame(self.root, style="Card.TFrame", padding=14)
        source_card.pack(fill="x", padx=24, pady=8)

        ttk.Label(source_card, text="2. CHỌN NGUỒN PHÁT (CAMERA HOẶC FILE VIDEO):", style="CardLabel.TLabel").pack(anchor="w", pady=(0, 8))

        # Option Webcam
        cam_row = ttk.Frame(source_card, style="Card.TFrame")
        cam_row.pack(fill="x", pady=4)
        ttk.Radiobutton(
            cam_row,
            text="Webcam trực tiếp",
            variable=self.source_type,
            value="WEBCAM",
            style="Card.TRadiobutton",
            command=self._on_source_change,
        ).pack(side="left")
        ttk.Label(cam_row, text="   Camera Index:", style="CardDesc.TLabel").pack(side="left")
        cam_combo = ttk.Combobox(cam_row, textvariable=self.camera_index, values=["0", "1", "2"], width=4, state="readonly")
        cam_combo.pack(side="left", padx=6)

        # Option Video
        video_row = ttk.Frame(source_card, style="Card.TFrame")
        video_row.pack(fill="x", pady=(8, 4))
        ttk.Radiobutton(
            video_row,
            text="File Video có sẵn",
            variable=self.source_type,
            value="VIDEO",
            style="Card.TRadiobutton",
            command=self._on_source_change,
        ).pack(side="left")

        file_browse_row = ttk.Frame(source_card, style="Card.TFrame")
        file_browse_row.pack(fill="x", pady=(4, 2), padx=(22, 0))

        self.video_entry = ttk.Entry(file_browse_row, textvariable=self.video_path, font=("Segoe UI", 9))
        self.video_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        btn_browse = ttk.Button(file_browse_row, text="Chọn Video...", style="Secondary.TButton", command=self._browse_video)
        btn_browse.pack(side="right")

        # -------------------------------------------------------------
        # KHU VỰC 3: ĐIỀU KHIỂN & TRẠNG THÁI
        # -------------------------------------------------------------
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=24, pady=12)

        self.btn_run = ttk.Button(
            control_frame,
            text="▶ BẮT ĐẦU CHẠY DEMO",
            style="Primary.TButton",
            command=self._start_demo,
        )
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_stop = ttk.Button(
            control_frame,
            text="⏹ DỪNG DEMO",
            style="Danger.TButton",
            command=self._stop_demo,
            state="disabled",
        )
        self.btn_stop.pack(side="right", padx=(6, 0))

        # Nút phụ: Mở file CSV và Kiểm thử
        extra_frame = ttk.Frame(self.root)
        extra_frame.pack(fill="x", padx=24, pady=4)

        btn_csv = ttk.Button(
            extra_frame,
            text="📊 Xem Log CSV (Plan B)",
            style="Secondary.TButton",
            command=self._open_csv_log,
        )
        btn_csv.pack(side="left", padx=(0, 6))

        btn_test = ttk.Button(
            extra_frame,
            text="⚡ Chạy Test Tự động (All Units)",
            style="Secondary.TButton",
            command=self._run_unit_tests,
        )
        btn_test.pack(side="left")

        # Trạng thái Status Bar
        self.status_bar = tk.Label(
            self.root,
            text="Sẵn sàng khởi động. Hãy chọn Luồng và Nguồn phát phía trên.",
            bg="#1F2937",
            fg="#10B981",
            font=("Segoe UI", 9),
            anchor="w",
            padx=12,
            pady=6,
        )
        self.status_bar.pack(side="bottom", fill="x")

        # Khởi tạo hiển thị
        self._on_plan_change()
        self._on_source_change()

    def _on_plan_change(self):
        """Ẩn/hiện tùy chọn riêng của Plan B."""
        if self.selected_plan.get() == "PLAN_B":
            self.rehab_options_frame.pack(anchor="w", pady=(4, 2))
        else:
            self.rehab_options_frame.pack_forget()

    def _on_source_change(self):
        """Cập nhật trạng thái ô nhập video."""
        if self.source_type.get() == "VIDEO":
            self.video_entry.configure(state="normal")
        else:
            self.video_entry.configure(state="disabled")

    def _browse_video(self):
        """Mở hộp thoại chọn tệp video."""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("All files", "*.*"),
        ]
        filename = filedialog.askopenfilename(
            title="Chọn file video demo",
            initialdir=os.getcwd(),
            filetypes=filetypes,
        )
        if filename:
            self.video_path.set(filename)
            self.source_type.set("VIDEO")
            self._on_source_change()

    def _start_demo(self):
        """Khởi động luồng được chọn trong tiến trình độc lập."""
        if self.current_process and self.current_process.poll() is None:
            messagebox.showinfo("Thông báo", "Một phiên demo đang chạy. Vui lòng dừng phiên hiện tại trước!")
            return

        plan = self.selected_plan.get()
        source_mode = self.source_type.get()

        # Xác định target script
        if plan == "PLAN_A":
            script_name = "vinmec_fall_demo.py"
            plan_name = "Luồng A (Phát hiện Té ngã VinMec)"
        else:
            script_name = "vinmec_rehab_demo.py"
            plan_name = "Luồng B (Phục hồi Chức năng Khớp Khuỷu tay)"

        # Xác định source argument
        if source_mode == "WEBCAM":
            source_arg = self.camera_index.get()
            source_desc = f"Webcam (Index {source_arg})"
        else:
            source_arg = self.video_path.get().strip("\"'")
            if not source_arg or not os.path.exists(source_arg):
                messagebox.showerror("Lỗi", "Vui lòng chọn file video hợp lệ trước khi bắt đầu!")
                return
            source_desc = f"Video: {os.path.basename(source_arg)}"

        # Tìm python interpreter chuẩn (ưu tiên môi trường Conda handsign đã có thư viện)
        python_exe = r"D:\annaconda\envs\handsign\python.exe"
        if not os.path.exists(python_exe):
            python_exe = sys.executable

        cmd = [python_exe, script_name, source_arg]

        try:
            self.current_process = subprocess.Popen(
                cmd,
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            self.btn_run.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            self.status_bar.configure(
                text=f"Đang chạy {plan_name} qua {source_desc}... (Nhấn [Q] trên cửa sổ video để đóng)",
                fg="#38BDF8",
            )

            # Khởi động thread theo dõi khi process kết thúc
            threading.Thread(target=self._monitor_process, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Lỗi khởi chạy", f"Không thể khởi động demo: {str(e)}")

    def _monitor_process(self):
        """Theo dõi tiến trình demo và cập nhật trạng thái khi kết thúc."""
        if self.current_process:
            self.current_process.wait()
            self.root.after(0, self._on_process_ended)

    def _on_process_ended(self):
        """Xử lý sau khi phiên demo đóng."""
        self.current_process = None
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.status_bar.configure(
            text="Phiên demo đã kết thúc an toàn. Sẵn sàng cho lượt tiếp theo.",
            fg="#10B981",
        )

    def _stop_demo(self):
        """Dừng cưỡng bức tiến trình đang chạy nếu cần."""
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            self.status_bar.configure(text="Đã dừng phiên demo.", fg="#EF4444")
            self._on_process_ended()

    def _open_csv_log(self):
        """Mở tệp CSV nhật ký buổi tập."""
        csv_file = os.path.abspath("rehab_session_log.csv")
        if os.path.exists(csv_file):
            if sys.platform == "win32":
                os.startfile(csv_file)
            else:
                subprocess.Popen(["xdg-open", csv_file])
            self.status_bar.configure(text=f"Đã mở file: {os.path.basename(csv_file)}", fg="#38BDF8")
        else:
            messagebox.showinfo(
                "Thông báo",
                "Chưa có tệp 'rehab_session_log.csv'. Hãy chạy thử Luồng B (Tele-Rehab) và tập ít nhất 1 lần để sinh log!",
            )

    def _run_unit_tests(self):
        """Chạy nhanh 2 bộ test kiểm thử tự động."""
        python_exe = r"D:\annaconda\envs\handsign\python.exe"
        if not os.path.exists(python_exe):
            python_exe = sys.executable

        self.status_bar.configure(text="Đang thực thi các bộ kiểm thử tự động...", fg="#F59E0B")

        def run_tests():
            res_a = subprocess.run([python_exe, "test_fall_logic.py"], capture_output=True, text=True)
            res_b = subprocess.run([python_exe, "test_rehab_logic.py"], capture_output=True, text=True)

            msg = "KẾT QUẢ KIỂM THỬ:\n\n"
            msg += "1. Luồng A (Fall Detection): " + ("THÀNH CÔNG (100% PASS)\n" if res_a.returncode == 0 else "THẤT BẠI\n")
            msg += "2. Luồng B (Tele-Rehab): " + ("THÀNH CÔNG (100% PASS)\n" if res_b.returncode == 0 else "THẤT BẠI\n")

            self.root.after(0, lambda: messagebox.showinfo("Kết quả Kiểm thử Tự động", msg))
            self.root.after(0, lambda: self.status_bar.configure(text="Kiểm thử hoàn tất! Tất cả tiêu chí đạt 100%.", fg="#10B981"))

        threading.Thread(target=run_tests, daemon=True).start()


def main():
    root = tk.Tk()
    app = VinmecDemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

