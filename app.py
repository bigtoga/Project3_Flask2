from flask import Flask, request, jsonify, make_response, json, Response
from pycaret.classification import *
import pickle
import numpy as np
import pandas as pd
import logging


logging.basicConfig(level=logging.DEBUG)

# Bug in Flask - have to add these two lines before importing flask_restplus
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property

# Note that flask_restplus requires classes for all routes, not just functions
from flask_restplus import Api, Resource, fields

app = Flask(__name__)

api = Api(app = app, 
		  version = "1.0", 
		  title = "Project3 Flask App", 
		  description = "Predict results using a trained model")

# PyCaret automatically adds ".pkl"
model_file = 'models/et20200526_2010'

# Holdout data
holdoutData = 'data/holdout-2012.csv'

# Random generated data
baseFolder = "data/RandomSamples"

# Target column
target = 'zeroBalCode'

# List the features in the model
feature_cols = [
    'origChannel'
    , 'loanPurp'
    , 'bankNumber'
    , 'stateNumber'
    , 'mSA'
    , 'origIntRate'
    , 'origUPB'
    , 'origLTV'
    , 'numBorrowers'
    , 'origDebtIncRatio'
    , 'worstCreditScore'
]

predict_fields = api.model('Predict Inputs', {
    'howManyFiles': fields.Integer(
			required = True
			, description="How many files should we generate?"
			, help="Required field"
			, min=1
			, max=10
		)
	, 'howManyRows': fields.Integer(
			required = True
			, description="How many rows in each file?"
			, help="Required field"
			, min=10
			, max=100
		)
	, 'fileName': fields.String(
			required = True
			, description="What should the files start with?"
			, help="Required field"
		)
    , 'randomFile': fields.String(
			required = True
			, description="Of the generated files, which one do you choose?"
			, help="Required field"
		)
})

#########################################################
# Pass in the name of the file selected and the model will
#    predict for you
#########################################################
# @app.route('/predict',methods=['POST'])
# @return_json
@api.route("/predict", methods=['OPTIONS','POST'])
class PredictClass(Resource):
	
#########################################################    
# Pass in the answers and this method will create the 
#    test files from the holdout data
#########################################################
	@api.header('Access-Control-Allow-Origin', '*')
	@api.header('Access-Control-Allow-Headers', '*')
	@api.header('Access-Control-Allow-Methods', '*')
	def get(self):
		response = Response({ \
				"status": "Error - GET not implemented"
				, "info": "Using POST, pass randomFile" \
			} \
			, mimetype='application/json' \
		)
		response.headers["Access-Control-Allow-Origin"] = "*"
		response.headers["Access-Control-Allow-Headers"] = "*"
		response.headers["Access-Control-Allow-Methods"] = "*"
		return response

#	@api.marshal_list_with(predict_fields)
	@api.doc(body=predict_fields)
	#@api.doc(parser=parser_randomFile)
	# @api.param('randomFile', 'The name of your file')
	#@api.doc(responses={404: 'randomFile not found'}, params={'randomFile': 'Name of the .csv file (MyFile01.csv)'})
	#@crossdomain(origin='*')
	# React submits a POST but Flask sees OPTIONS
	@api.header('Access-Control-Allow-Origin', '*')
	@api.header('Access-Control-Allow-Headers', '*')
	@api.header('Access-Control-Allow-Methods', '*')
	def options(self):
		app.logger.info('/predict called')

		try:
			formData = request.json
			app.logger.error(f'/predict called with formData: {formData}')
			app.logger.error(f'/predict called with request: {request}')

			howManyFiles = formData["howManyFiles"]
			howManyRows = formData["howManyRows"]
			fileName = formData["fileName"]
			randomFile = formData["randomFile"]
		except Exception as error:
			response = jsonify({ \
				"statusCode": 404  \
				, "error": str(error)  \
			})
			return response

		try:
			howManyFiles = int(howManyFiles)
		except Exception as error:
			response = jsonify({ \
				"statusCode": 500  \
				, "detail": "howManyFiles is not an integer"  \
				, "error": str(error)  \
			})
			
			return response
		
		if howManyFiles < 1 or howManyFiles > 10:
			response = jsonify({ \
				"statusCode": 500  \
				, "error": "howManyFiles must be between 1 and 10"  \
			})
			
			return response
			
		try:
			howManyRows = int(howManyRows)
		except Exception as error:
			response = jsonify({ \
				"statusCode": 500  \
				, "detail": "howManyRows is not an integer"  \
				, "error": str(error)  \
			})
			
			return response
		
		if howManyRows < 1 or howManyRows > 10:
			response = jsonify({ \
				"statusCode": 500  \
				, "error": "howManyRows must be between 1 and 10"  \
			})
			
			return response
		
		if len(fileName) < 3 or len(fileName) > 16:
			response = jsonify({ \
				"statusCode": 500  \
				, "error": "fileName must be between 3 and 16 characters"  \
			})
			
			return response

		#########################################################
		# Generate the random files
		#########################################################
		try: 
			# Read in the holdout data into a Pandas DataFrame
			df = pd.read_csv(holdoutData)
		except: 
			response = jsonify({ \
				"statusCode": 500  \
				, "error": f"Unable to read holdoutData {holdoutData}" \
			})
			
			return response

		# Create the files withstarting with 01
		i = 1
		generatedFiles = []

		while i <= howManyFiles:
			# Make sortable filenames (01, 02, 03 instead of 1, 2, 3)
			namingNumber = "01"

			if i < 10:
				namingNumber = "0" + str(i)
			else:
				namingNumber = str(i)
			
			the_file = f'{baseFolder}/{fileName}{namingNumber}.csv'

			try:
			# Step 1: Let's delete any previous runs' files first:
				os.remove(the_file)
			except:
				pass # How to do an empty except in Python

			# So we can pass what the names of the files created are back to front end
			file_dict = {}
			file_dict["fileName"] = the_file
			generatedFiles.append(file_dict)

			# Export to csv
			df.sample(howManyRows).to_csv(the_file)

			# Get the next file or exit if processed last requested file
			i = i+1

		try:
			data_unseen = pd.read_csv(f'{baseFolder}/{randomFile}')
		except:			
			response = jsonify({ \
				"statusCode": 500  \
				, "error": f"File {randomFile} doesn't exist" \
			})
			
			return response

		# Load the model w PyCaret
		model = load_model(model_file)
		dfPredictions = predict_model(model, data=data_unseen)
		# prediction = int(prediction.Label[0])    
		# output = prediction.Label[0]

		# Remove the previous index columns
		# Don't drop both though - React requires a unique "key" for each row
		dfPredictions.drop(['Unnamed: 0'], 1, inplace=True)
		# dfPredictions.rename(columns = {'Unnamed: 0':'idx'}, inplace = True) 
		dfPredictions.drop(['Unnamed: 0.1'], 1, inplace=True)

		# response = json.dumps(dfPredictions.to_json(), 200, {'content-type': 'application/json'})
		# response = jsonify({'data': {dfPredictions.to_json(orient='records')}})
		# return Response(dfPredictions.to_json(orient="records"), mimetype='application/json')

		response = Response(dfPredictions.to_json(orient="table"), mimetype='application/json')
		response.headers["Access-Control-Allow-Origin"] = "*"
		response.headers["Access-Control-Allow-Headers"] = "*"
		response.headers["Access-Control-Allow-Methods"] = "*"
		return response


#api.add_resource(PredictClass, '/predict/<string:randomFile>')

if __name__ == '__main__':
    app.run(debug=True)