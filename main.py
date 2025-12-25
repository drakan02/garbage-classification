import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import cv2
import os
import tkinter as tk
from PIL import ImageTk

try:
    from model_handler import TrashClassifier
except ImportError:
    TrashClassifier = None 

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

ASSETS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")

COLORS = {
    "primary": "#00E676",      
    "primary_hover": "#00C853", 
    "danger": "#FF5252",       
    "danger_hover": "#FF1744",
    "info": "#2979FF",         
    "info_hover": "#2962FF",
    "dark_bg": "#121212",      
    "sidebar": "#1E1E1E",      
    "card": "#2D2D2D",         
    "nav_hover": "#333333",
    "text": "#FFFFFF",
    "subtext": "#B0B0B0"
}

FONTS = {
    "h1": ("Roboto", 32, "bold"),
    "h2": ("Roboto", 24, "bold"),
    "h3": ("Roboto", 24, "bold"),
    "body": ("Roboto", 16),
    "sidebar": ("Roboto", 14),
    "body_bold": ("Roboto", 20, "bold"),
    "btn": ("Roboto", 15, "bold"),
    "small": ("Roboto", 12)
}

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("AI Garbage Classification")
        self.geometry("1280x800")
        self.resizable(False, False)
        self.camera_loop_id = None
        self.cam_image = None
        
        self.ai = None
        if TrashClassifier:
            try:
                self.ai = TrashClassifier("best.pt")
            except:
                print("Model loading failed.")
        self.icons = self.load_assets()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_sidebar()

        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=40, pady=30)
        self.main_area.grid_columnconfigure(0, weight=1)
        self.main_area.grid_rowconfigure(0, weight=1)

        self.create_views()
        
        self.cap = None
        self.camera_running = False
        self.last_result = "No Data"
        self.show_frame("guide")
        self.update_status_panel()

    def load_assets(self):
        icons = {}
        
        def get_img_normalized(filename, content_size, target_size=(48, 48)):
            path = os.path.join(ASSETS_PATH, filename)
            if os.path.exists(path):
                img = Image.open(path).resize(content_size, Image.Resampling.LANCZOS)

                bg = Image.new("RGBA", target_size, (0, 0, 0, 0))

                left = (target_size[0] - content_size[0]) // 2
                top = (target_size[1] - content_size[1]) // 2

                bg.paste(img, (left, top), img)
                
                return ctk.CTkImage(bg, size=target_size)
            return None

        path_logo = os.path.join(ASSETS_PATH, "logo.png")
        if os.path.exists(path_logo):
            icons["logo"] = ctk.CTkImage(Image.open(path_logo), size=(100, 100))
        else:
            icons["logo"] = None
        
        icons["home"] = get_img_normalized("home.png", content_size=(32, 32)) 
        icons["upload"] = get_img_normalized("upload.png", content_size=(48, 48)) 
        icons["camera"] = get_img_normalized("camera.png", content_size=(40, 40)) 
        
        def get_img_simple(filename, size):
            path = os.path.join(ASSETS_PATH, filename)
            return ctk.CTkImage(Image.open(path), size=size) if os.path.exists(path) else None

        icons["check"] = get_img_simple("check.png", (20, 20))
        icons["error"] = get_img_simple("error.png", (60, 60))
        icons["target"] = get_img_simple("target.png", (60, 60))
        
        return icons

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0, fg_color=COLORS["sidebar"])
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        self.fr_logo = ctk.CTkFrame(self.sidebar, fg_color="transparent", height=150)
        self.fr_logo.pack(fill="x", pady=(40, 20))
        
        if self.icons["logo"]:
            ctk.CTkLabel(self.fr_logo, text="", image=self.icons["logo"]).pack()
        else:
            ctk.CTkLabel(self.fr_logo, text="AI", font=("Roboto", 40, "bold")).pack()

        ctk.CTkLabel(self.fr_logo, text="AI GARBAGE\nCLASSIFICATION", font=FONTS["h3"], text_color=COLORS["primary"]).pack(pady=10)

        self.fr_nav = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.fr_nav.pack(fill="x", padx=15, pady=20)

        self.btn_guide = self.create_sidebar_btn("User Guide", "home", "guide")
        self.btn_upload = self.create_sidebar_btn("Upload Image", "upload", "upload")
        self.btn_camera = self.create_sidebar_btn("Live Camera", "camera", "camera")

        self.fr_stats = ctk.CTkFrame(self.sidebar, fg_color=COLORS["card"], corner_radius=15)
        self.fr_stats.pack(side="bottom", fill="x", padx=20, pady=16)
        
        ctk.CTkLabel(self.fr_stats, text="SYSTEM STATUS", font=FONTS["small"], text_color=COLORS["subtext"]).pack(pady=(15,5))
        
        self.lbl_model_status = ctk.CTkLabel(self.fr_stats, text="● Model: Checking...", font=FONTS["body"])
        self.lbl_model_status.pack(anchor="w", padx=20, pady=2)
        
        self.lbl_cam_status = ctk.CTkLabel(self.fr_stats, text="● Camera: OFF", font=FONTS["body"], text_color=COLORS["danger"])
        self.lbl_cam_status.pack(anchor="w", padx=20, pady=2)

        ctk.CTkFrame(self.fr_stats, height=1, fg_color="gray30").pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(self.fr_stats, text="LATEST RESULT:", font=FONTS["small"], text_color=COLORS["subtext"]).pack(anchor="w", padx=20)
        self.lbl_last_result = ctk.CTkLabel(self.fr_stats, text="---", font=FONTS["h3"], text_color=COLORS["primary"], wraplength=220)
        self.lbl_last_result.pack(pady=(0, 20))

    def create_sidebar_btn(self, text, icon_key, name):
        btn = ctk.CTkButton(
            self.fr_nav, 
            text=text,                   
            height=60,                  
            font=FONTS["btn"], 
            anchor="w", 
            fg_color="transparent", 
            corner_radius=10, 

            border_spacing=10,           
            
            hover_color=COLORS["nav_hover"],
            text_color=COLORS["text"],
            image=self.icons.get(icon_key), 
            compound="left",
            command=lambda: self.show_frame(name)
        )
        btn.pack(fill="x", pady=5)
        return btn

    def create_views(self):
        self.fr_guide = ctk.CTkFrame(self.main_area, fg_color="transparent")
        ctk.CTkLabel(self.fr_guide, text="Welcome Back!", font=FONTS["h1"]).pack(pady=(0, 25), anchor="w")
        
        guide_card = ctk.CTkFrame(self.fr_guide, fg_color=COLORS["card"], corner_radius=15)
        guide_card.pack(fill="both", expand=True, ipady=10)
        
        ctk.CTkLabel(guide_card, text="USER GUIDE & TROUBLESHOOTING", 
                     font=FONTS["h3"], text_color=COLORS["primary"]).pack(anchor="w", pady=(25, 20), padx=30)
        
        self.create_guide_row(guide_card, "error", "Black Screen:", "Ensure camera permissions are granted.")
        self.create_guide_row(guide_card, "target", "No Detection:", "Move object closer (30-50cm).")
        
        ctk.CTkFrame(guide_card, height=1, fg_color="#888").pack(fill="x", pady=20, padx=30) 
        
        ctk.CTkLabel(guide_card, text="SUPPORTED CATEGORIES:", 
                     font=FONTS["body_bold"], text_color=COLORS["text"]).pack(anchor="w", pady=(0, 15), padx=30)
        
        list_frame = ctk.CTkFrame(guide_card, fg_color="transparent")
        list_frame.pack(fill="both", padx=30, expand=True)
        
        support_items = [
            ("Battery", "AA, AAA, Lithium..."), ("Biological", "Food scraps..."),
            ("Cardboard", "Boxes..."), ("Clothes", "Shirts, pants..."),
            ("Glass", "Bottles..."), ("Metal", "Cans..."),
            ("Paper", "Newspapers..."), ("Plastic", "Bottles..."),
            ("Shoes", "Sneakers..."), ("Trash", "Non-recyclable...")
        ]
        
        for i, (title, desc) in enumerate(support_items):
            item_f = ctk.CTkFrame(list_frame, fg_color="transparent")
            item_f.grid(row=i//2, column=i%2, sticky="w", pady=5, padx=(0, 250))
            if self.icons["check"]:
                ctk.CTkLabel(item_f, text="", image=self.icons["check"]).pack(side="left", padx=(0, 10))
            ctk.CTkLabel(item_f, text=title, font=FONTS["body_bold"], text_color="white").pack(side="left")
            ctk.CTkLabel(item_f, text=f" - {desc}", font=FONTS["body"], text_color="gray").pack(side="left")

        self.fr_upload = ctk.CTkFrame(self.main_area, fg_color="transparent")
        ctk.CTkLabel(self.fr_upload, text="Static Image Analysis", font=FONTS["h2"]).pack(pady=(0, 20), anchor="center")
        
        self.img_container = ctk.CTkFrame(self.fr_upload, fg_color="black", corner_radius=15, border_width=2, border_color="#333")
        self.img_container.pack(fill="both", expand=True, pady=10)
        
        self.lbl_up_img = ctk.CTkLabel(self.img_container, text="[Select Image]", text_color="gray", font=FONTS["body"])
        self.lbl_up_img.place(relx=0.5, rely=0.5, anchor="center")
        
        ctrl_frame = ctk.CTkFrame(self.fr_upload, fg_color="transparent")
        ctrl_frame.pack(fill="x", pady=20)
        
        ctk.CTkButton(ctrl_frame, text="SELECT IMAGE", height=50, width=200, 
                      fg_color=COLORS["primary"], text_color=COLORS["card"], hover_color=COLORS["primary_hover"],
                      font=FONTS["btn"], image=self.icons.get("upload"), 
                      command=self.open_image).pack(side="bottom")
        self.lbl_up_res = ctk.CTkLabel(ctrl_frame, text="", font=FONTS["h2"], text_color=COLORS["primary"])
        self.lbl_up_res.pack(side="right", padx=20)

        self.fr_camera = ctk.CTkFrame(self.main_area, fg_color="transparent")

        self.cam_container = ctk.CTkFrame(
            self.fr_camera,
            fg_color="black",
            corner_radius=15,
            border_width=2,
            border_color="#444"
        )
        self.cam_container.pack(fill="both", expand=True, pady=(0, 20))

        # Label chỉ để HIỂN THỊ ẢNH
        self.lbl_cam = ctk.CTkLabel(self.cam_container, text="")
        self.lbl_cam.pack(fill="both", expand=True)

        # Label RIÊNG cho trạng thái OFF
        self.lbl_cam_text = ctk.CTkLabel(
            self.cam_container,
            text="Camera OFF",
            text_color="gray",
            font=FONTS["body"]
        )
        self.lbl_cam_text.place(relx=0.5, rely=0.5, anchor="center")

        self.btn_cam_toggle = ctk.CTkButton(
            self.fr_camera,
            text="START CAMERA",
            height=55,
            fg_color=COLORS["primary"],
            text_color="black",
            hover_color=COLORS["primary_hover"],
            font=FONTS["btn"],
            image=self.icons.get("camera"),
            command=self.toggle_cam
        )
        self.btn_cam_toggle.pack(fill="x")

    def create_guide_row(self, parent, icon_key, title, text):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", pady=8, padx=30)
        if self.icons.get(icon_key):
             ctk.CTkLabel(f, text="", image=self.icons[icon_key]).pack(side="left", anchor="n", pady=2)
        content_f = ctk.CTkFrame(f, fg_color="transparent")
        content_f.pack(side="left", fill="x", expand=True, padx=(15, 0))
        ctk.CTkLabel(content_f, text=title, font=FONTS["body_bold"], text_color="white").pack(anchor="w")
        ctk.CTkLabel(content_f, text=text, font=FONTS["body"], text_color=COLORS["subtext"]).pack(anchor="w")

    def update_status_panel(self):
        if self.ai and getattr(self.ai, 'model', None):
            self.lbl_model_status.configure(text="● Model: Ready", text_color=COLORS["primary"])
        else:
            self.lbl_model_status.configure(text="● Model: Error", text_color=COLORS["danger"])
        self.lbl_last_result.configure(text=self.last_result)

    def set_btn_active(self, btn, is_active):
        if is_active:
            btn.configure(fg_color=COLORS["primary"], text_color="black", hover_color=COLORS["primary_hover"])
        else:
            btn.configure(fg_color="transparent", text_color="white", hover_color=COLORS["nav_hover"])

    def show_frame(self, name):
        self.fr_guide.pack_forget()
        self.fr_upload.pack_forget()
        self.fr_camera.pack_forget()
        
        self.set_btn_active(self.btn_guide, False)
        self.set_btn_active(self.btn_upload, False)
        self.set_btn_active(self.btn_camera, False)
        
        if name == "guide": 
            self.fr_guide.pack(fill="both", expand=True)
            self.set_btn_active(self.btn_guide, True)
        elif name == "upload": 
            self.fr_upload.pack(fill="both", expand=True)
            self.set_btn_active(self.btn_upload, True)
        elif name == "camera": 
            self.fr_camera.pack(fill="both", expand=True)
            self.set_btn_active(self.btn_camera, True)
            
        if name != "camera":
            if self.camera_running:
                self.toggle_cam()   # TẮT THỰC SỰ

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg;*.png;*.jpeg")])
        if path:
            img = Image.open(path)
            display_img = img.copy()
            display_img.thumbnail((800, 600)) 
            self.lbl_up_img.configure(image=ctk.CTkImage(display_img, size=display_img.size), text="")
            self.lbl_up_img.place(relx=0.5, rely=0.5, anchor="center")
            if self.ai:
                label, conf = self.ai.predict_frame(cv2.imread(path))
                self.last_result = f"{label.upper()}\n({conf:.1%})"
                self.lbl_up_res.configure(text=f"RESULT: {label.upper()} ({conf:.1%})")
                self.update_status_panel()

    def toggle_cam(self):
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open camera")
                return

            self.camera_running = True

            self.btn_cam_toggle.configure(
                text="STOP CAMERA",
                fg_color=COLORS["danger"],
                hover_color=COLORS["danger_hover"],
                text_color="white"
            )

            self.lbl_cam_status.configure(
                text="● Camera: ON",
                text_color=COLORS["primary"]
            )

            self.lbl_cam_text.place_forget()
            self.cam_container.configure(border_color=COLORS["primary"])

            # 🔒 TẠO 1 CTkImage DUY NHẤT
            self.ctk_cam_image = ctk.CTkImage(
                light_image=Image.new("RGB", (640, 480)),
                dark_image=Image.new("RGB", (640, 480)),
                size=(640, 480)
            )
            self.lbl_cam.configure(image=self.ctk_cam_image)

            self.loop_camera()

        else:
            self.camera_running = False

            if self.camera_loop_id:
                self.after_cancel(self.camera_loop_id)
                self.camera_loop_id = None

            if self.cap:
                self.cap.release()
                self.cap = None

            # ✅ RESET IMAGE HOÀN TOÀN
            black_img = Image.new("RGB", (640, 480), "black")
            self.ctk_cam_image.configure(
                light_image=black_img,
                dark_image=black_img
            )

            # ❗ XÓA REFERENCE
            self.ctk_cam_image = None
            self.lbl_cam.configure(image=None)

            # HIỆN TEXT OFF
            self.lbl_cam_text.place(relx=0.5, rely=0.5, anchor="center")

            self.btn_cam_toggle.configure(
                text="START CAMERA",
                fg_color=COLORS["primary"],
                hover_color=COLORS["primary_hover"],
                text_color="black"
            )

            self.lbl_cam_status.configure(
                text="● Camera: OFF",
                text_color=COLORS["danger"]
            )

            self.cam_container.configure(border_color="#444")


    def loop_camera(self):
        if not self.camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.toggle_cam()
            return

        frame = cv2.flip(frame, 1)

        if self.ai:
            label, conf = self.ai.predict_frame(frame)
            if conf > 0.5:
                self.last_result = f"{label.upper()} ({conf:.1%})"
                self.lbl_last_result.configure(text=self.last_result)

            cv2.putText(
                frame,
                f"{label.upper()} ({conf:.0%})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((640, 480))

        # ✅ CHỈ UPDATE NỘI DUNG
        self.ctk_cam_image.configure(
            light_image=img,
            dark_image=img
        )

        self.camera_loop_id = self.after(30, self.loop_camera)


if __name__ == "__main__":
    app = App()
    app.mainloop()