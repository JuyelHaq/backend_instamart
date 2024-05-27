from flask import request, jsonify
import os

BASE_UPLOAD_FOLDER = "/home/clients"  # Update with your actual base upload folder

@app.before_request
def before_request():
    response_data = {"success": False, "message": "", "user_directory_path": None}

    token = request.headers.get('Authorization')
    if token and token.startswith('Bearer '):
        actual_token = token.split('Bearer ')[1]
    else:
        # Handle the case when the token is not in the expected format
        response_data["message"] = "Invalid token format"
        return jsonify(response_data)

    try:
        # Retrieve the user's ID from the Customers table based on the token
        get_user_id_query = 'SELECT user_id FROM Customers WHERE api_token = %s'
        cursor.execute(get_user_id_query, (actual_token,))
        user_id_result = cursor.fetchone()

        if user_id_result:
            user_id = user_id_result[0]

            # Retrieve the user's directory name from the Directory table
            get_directory_query = 'SELECT directory_name FROM Directory WHERE user_id = %s'
            cursor.execute(get_directory_query, (user_id,))
            directory_info = cursor.fetchone()

            if directory_info:
                directory_name = directory_info[0]
                user_directory_path = os.path.join(BASE_UPLOAD_FOLDER, directory_name)
                response_data["success"] = True
                response_data["message"] = "File uploaded successfully"
                response_data["user_directory_path"] = user_directory_path
                print(user_directory_path)
            else:
                response_data["message"] = "User directory not found"
        else:
            response_data["message"] = "Invalid token"
    except Exception as e:
        print(f"Error: {e}")
        response_data["message"] = "An error occurred while processing the request"

    return jsonify(response_data)

