from flask import Flask, render_template
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__, 
                static_folder='static',
                template_folder='templates')
    CORS(app)
    
    # Load configuration
    app.config.from_object('api.config.database')
    
    # Register blueprints
    from api.handlers.qa_handler import qa_bp
    app.register_blueprint(qa_bp, url_prefix='/api')
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/health')
    def health_check():
        return {'status': 'ok', 'message': 'Medical QA System is running'}
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)