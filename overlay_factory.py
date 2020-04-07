import image_processor

class OverlayFactory(object):
    def __init__(self):
        self.overlay_queue = []
    
    def set_base_image(self, img):
        self.overlay_queue = []
        self.overlay_queue.append((img, None))
    
    def add_overlay(self, img, pos):
        self.overlay_queue.append((img, pos))
    
    def produce_overlay(self):
        base_image = self.overlay_queue[0][0]

        if len(self.overlay_queue) == 0:
            return None
        if len(self.overlay_queue) == 1:
            return base_image

        processor = image_processor.ImageProcessor()
        combined = processor.overlay_images(base_image, self.overlay_queue[1][0], self.overlay_queue[1][1])

        for i in range(2, len(self.overlay_queue)):
            combined = processor.overlay_images(combined, self.overlay_queue[i][0], self.overlay_queue[i][1])
        
        return combined
