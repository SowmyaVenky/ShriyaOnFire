import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import os

# Configuration
port = 587
smtp_server = "live.smtp.mailtrap.io"
login = "api"  # Your login generated by Mailtrap
password = "5f61752a6ca796c82afc51eb5c809495"  # Your password generated by Mailtrap

sender_email = "shriyaonfire@demomailtrap.com"
receiver_email = "shriyaonfire@gmail.com"

# HTML content with an image embedded
html = """\
<html>
  <body>
    <p>FIRE ALERT!!!<br>
    This is a <b>fire alert</b> email with an embedded image.<br>
    <img src="cid:image1">.</p>
  </body>
</html>
"""

# Create a multipart message and set headers
message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = "Fire Detected!!!"

# Attach the HTML part
message.attach(MIMEText(html, "html"))

# Specify the path to your image
image_path = "C:/Venky/Shriya_on_Fire/ShriyaOnFire/image_extractions/gas_images/valid/images/461af0d3-c985-4e78-9a68-240f3fd55151_frame2400_jpg.rf.7e853d5a46560b64997c30dcaa6c8c51.jpg"  # Change this to the correct path

# Open the image file in binary mode
with open(image_path, 'rb') as img:
    # Attach the image file
    msg_img = MIMEImage(img.read(), name=os.path.basename(image_path))
    # Define the Content-ID header to use in the HTML body
    msg_img.add_header('Content-ID', '<image1>')
    # Attach the image to the message
    message.attach(msg_img)

# Send the email
with smtplib.SMTP(smtp_server, port) as server:
    server.starttls()
    server.login(login, password)
    server.sendmail(sender_email, receiver_email, message.as_string())

print('Sent')
