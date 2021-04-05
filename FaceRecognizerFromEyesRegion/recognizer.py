MAXIMUM_DISTANCE_THRESHOLD = 1.400.000
RECONSTRUCTION_THRESHOLD = 33.0

def recognize(error_values):
  error = error_values[0]
  error_k = error_values[1][1]
  if(error_k < MAXIMUM_DISTANCE_THRESHOLD and error < RECONSTRUCTION_THRESHOLD):
    return error_values[1][0] #index k
  elif(error_k >= MAXIMUM_DISTANCE_THRESHOLD and error < RECONSTRUCTION_THRESHOLD):
    return "Unkown face"
  elif(error >= RECONSTRUCTION_THRESHOLD):
    return "Not a human face"