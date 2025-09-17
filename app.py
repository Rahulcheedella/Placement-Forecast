import joblib
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder='student_template')

curr_model = joblib.load("Current_model.pkl")
exp_model = joblib.load("Expected_model.pkl")
label_encoder = joblib.load("Labels.pkl")
scaler = joblib.load("Scalers.pkl")

def predict_companies(data):
    data['skill_score'] = (
        data['problem_solving_skill'] * 0.4 + data['programming_skill'] * 0.3 + data['leetcode_score'] / 25 * 0.3
    )
    total_score = (
        data['skill_score'] * 0.5 + data['problem_solving_skill'] * 0.3 + data['programming_skill'] * 0.2 + 
        data['gpa'] * 0.1
    )

    if total_score > 8.5:
        curr_companies = ["Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix"]
        exp_companies = ["Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix"]
    elif total_score > 7.5:
        curr_companies = ["Adobe", "Salesforce", "Atlassian", "Nvidia", "Uber", "LinkedIn"]
        exp_companies = ["Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix"]
    elif total_score > 6.5:
        curr_companies = ["Zoho", "Freshworks", "Razorpay", "Postman", "Swiggy", "Zomato"]
        exp_companies = ["Adobe", "Salesforce", "Atlassian", "Nvidia", "Uber", "LinkedIn"]
    else:
        curr_companies = exp_companies = ["Infosys", "TCS", "Wipro", "HCL", "Accenture", "Cognizant"]

    curr_companies = np.random.choice(curr_companies, 3, replace = False)
    exp_companies = np.random.choice(exp_companies, 3, replace = False)

    return ' ,'.join(curr_companies), ' ,'.join(exp_companies)

def extraTime(data, curr_sal, exp_sal):
    skill_gap = max(0, (8 - data['programming_skill']) * 10 +
                       (180 - data['leetcode_score']) * 0.1)
    salary_gap = max(0, exp_sal - curr_sal)

    extra_hours = round(1 + (skill_gap/100 * 6) + (salary_gap/10 * 0.5), 1)
    return min(8, max(1, extra_hours))

@app.route('/')
def home():
    return render_template('student.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not all([curr_model, exp_model, label_encoder, scaler]):
        return jsonify({'error': 'Model artifacts not loaded. Please check the server.'}), 500
    try:
        data = request.get_json()
        new_data = pd.DataFrame([data])
        new_data['college_tier_encoded'] = label_encoder.transform(new_data['college_tier'])[0]
    
    
        feature_columns = ['gpa', 'college_tier_encoded', 'problem_solving_skill',
                       'programming_skill', 'leetcode_score', 'project_count',
                       'internship_months', 'skill_score']

        new_data['skill_score'] = (
        new_data['problem_solving_skill'] * 0.4 + new_data['programming_skill'] * 0.3 + new_data['leetcode_score'] / 25 * 0.3
        )

        x_input = new_data[feature_columns]
        x_scaled = scaler.transform(x_input)

        curr_sal = curr_model.predict(x_scaled)[0]
        exp_sal = exp_model.predict(x_scaled)[0]

        curr_salary = max(5, min(50, curr_sal))
        exp_salary = max(curr_sal, min(60, exp_sal))

        curr_companies, exp_companies = predict_companies(data)
        extra_time = extraTime(data, curr_salary, exp_salary)
        response =  {
        'current_Salary' : round(curr_salary,2),
        'expected_Salary' : round(exp_salary,2),
        'current_Companies' : curr_companies,
        'expected_Companies' : exp_companies,
        'Extra_Time_Required' : extra_time
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=True)