# Custom error handlers
from flask import jsonify

def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            "success": False,
            "error": "Resource not found",
            "code": 404
        }), 404

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({
            "success": False,
            "error": "Bad request",
            "code": 400,
            "message": str(e)
        }), 400

    @app.errorhandler(500)
    def internal_error(e):
        # Log error internally
        app.logger.error(f"Internal Server Error: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "code": 500
        }), 500

def custom_error(message, code=400):
    """Utility function to return JSON error from routes"""
    return jsonify({
        "success": False,
        "error": message,
        "code": code
    }), code
