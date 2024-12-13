

import json
class Config:
    def __init__(self):
        with open("config.json", "r") as f:
            self.config = json.loads(f.read())
        
    def get_property(self, property_name):
        if property_name not in self.config.keys():
            return None
        
        return self.config[property_name]