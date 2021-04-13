from pyfume import pyFUME
from LoadData import DataLoader
from Splitter import DataSplitter
from ModelBuilder import SugenoFISBuilder
from Clustering import Clusterer
from EstimateAntecendentSet import AntecedentEstimator
from EstimateConsequentParameters import ConsequentEstimator
from Tester import SugenoFISTester
#%%
# Set the path to the data and choose the number of clusters
path='C:/Users/cesar/Desktop/fuzzyPython/Concrete_Data.xls'
nc=3
#%%
# Load and normalize the data using min-max normalization
dl=DataLoader(path,normalize='minmax')
variable_names=dl.variable_names 
#%%
# Generate the Takagi-Sugeno FIS
FIS = pyFUME(datapath=path, nr_clus=nc)

# Calculate and print the accuracy of the generated model
MAE=FIS.calculate_error(method="MAE")
print ("The estimated error of the developed model is:", MAE)

## Use the FIS to predict the compressive strength of a new concrete sample
# Extract the model from the FIS object
model=FIS.get_model()

# Set the values for each variable
model.set_variable('Cement', 300.0)
model.set_variable('BlastFurnaceSlag', 50.0)
model.set_variable('FlyAsh', 0.0)
model.set_variable('Water', 175.0)
model.set_variable('Superplasticizer',0.7)
model.set_variable('CoarseAggregate', 900.0)
model.set_variable('FineAggregate', 600.0)
model.set_variable('Age', 45.0)

# Perform inference and print predicted value
print(model.Sugeno_inference(['OUTPUT']))