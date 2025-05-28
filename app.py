from flask import Flask, request, jsonify
from flask_cors import CORS
from ocp_solver import solve_ocp
import traceback

app = Flask(__name__)
CORS(app)

@app.route("/solve", methods=["POST"])
def solve():
    try:
        data = request.get_json()
        k = float(data.get("k", 1.0))
        result = solve_ocp(k)

        return jsonify({"success": True,
                        "data": result})
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500
    
if __name__ == "__main__":
    app.run(debug=True)