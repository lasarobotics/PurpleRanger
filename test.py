import depthai

class pipeline:
    def __init__(self, pipeline: depthai.Pipeline, table):
        cam = pipeline.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_A)
        self.queue = cam.out.createOutputQueue() #I know this is not correct but it gets the point accros

    def start(self):
        #do something with the cam data here
        while True:
            print(self.queue.get())

#main code

with depthai.Pipeline() as p:
    camera = pipeline(p, None)
    #This is also incorrect and threading will be needed
    camera.start()
    p.start()





