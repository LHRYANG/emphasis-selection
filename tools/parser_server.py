
import os
from nltk.parse.corenlp import CoreNLPServer

if __name__=='__main__':
    java_path = "C:/Program Files/Java/jdk1.8.0_231/bin/java.exe"
    os.environ['JAVAHOME'] = java_path


    # The server needs to know the location of the following files:
    #   - stanford-corenlp-X.X.X.jar
    #  - stanford-corenlp-X.X.X-models.jar

    STANFORD = "../stanford-corenlp-full-2018-10-05"
    # Create the server
    server = CoreNLPServer(
       os.path.join(STANFORD, "stanford-corenlp-3.9.2.jar"),
       os.path.join(STANFORD, "stanford-corenlp-3.9.2-models.jar"),
    )

    # Start the server in the background
    server.start()


