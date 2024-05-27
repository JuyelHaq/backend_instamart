from qa_over_docs import   app
from flask_cors import CORS
import ssl
#allowed_origin = 'gentai.instamart.ai'  
#CORS(app, origins=[allowed_origin])
CORS(app)

if __name__ == '__main__':
    ssl_cert = '/etc/letsencrypt/live/gentai.instamart.ai/fullchain.pem'
    ssl_key = '/etc/letsencrypt/live/gentai.instamart.ai/privkey.pem'
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile=ssl_cert, keyfile=ssl_key)
    app.run(debug=True, port=8446, host='gentai.instamart.ai',ssl_context=ssl_context)
                 #ssl_context=(ssl_cert, ssl_key))
    
