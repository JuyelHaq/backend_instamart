from flask import Flask
from flask_mail import Mail, Message

app = Flask(__name__)

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.zoho.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'juyel@instamart.ai'
app.config['MAIL_PASSWORD'] = 'Juyel257!'
app.config['MAIL_DEFAULT_SENDER'] = 'juyel@instamart.ai'

mail = Mail(app)

def send_email(recipient, subject, body):
    msg = Message(subject, recipients=[recipient])
    msg.body = body
    mail.send(msg)

if __name__ == '__main__':
    # Example usage:
    send_email('juyel@thirdeyedata.ai', 'Test Email', 'This is a test email.')

