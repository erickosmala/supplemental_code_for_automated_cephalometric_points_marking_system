
class ConfigProfile:

    def __init__(self):
        self.batchSize = 1
        self.landmarkNum = 19
        self.image_scale = (800, 640)
        self.use_gpu = 0
        self.spacing = 0.1
        self.R1 = 41
        self.R2 = 41
        self.epochs = 350
        self.data_enhanceNum = 1
        self.dataRoot = "data/"
        self.supervised_dataset_train = "images_to_process/images/"
        self.supervised_dataset_val = "images_to_process/images/"
        self.supervised_dataset_test = "images_to_process/images/"
        self.trainingSetCsv = "400_senior_train.csv"
        self.valSetCsv = "400_senior_val.csv"
        self.testSetCsv = "400_senior_test.csv"
