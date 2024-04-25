#Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data, var_name):

  c_str = ''

  # Create header guard
  c_str += '#ifndef ' + var_name.upper() + '_H\n'
  c_str += '#define ' + var_name.upper() + '_H\n\n'
#
  # Add array length at top of file
  c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

  # Declare C variable
  c_str += 'unsigned char ' + var_name + '[] = {'
  hex_array = []
  for i, val in enumerate(hex_data) :

    # Construct string from hex
    hex_str = format(val, '#04x')

    # Add formatting so each line stays within 80 characters
    if (i + 1) < len(hex_data):
      hex_str += ','
    if (i + 1) % 12 == 0:
      hex_str += '\n '
    hex_array.append(hex_str)

  # Add closing brace
  c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

  # Close out header guard
  c_str += '#endif //' + var_name.upper() + '_H'

  return c_str

'''
def tflite_to_hex(file_path):
    with open(file_path, 'rb') as file:
        tflite_data = file.read()
    hex_data = tflite_data.hex()
    return hex_data

# Example usage:
tflite_file_path = r"C:\Users\wengm\anaconda3\envs\mphy0043_cw_final\rnn_model.tflite"
tflite_model_hex = tflite_to_hex(tflite_file_path)
'''

# Write TFLite model to a C source (or header) file
c_model_name = "rnn_model.h"
tflite_model_hex = r"C:\Users\wengm\anaconda3\envs\mphy0043_cw_final\rnn_model_c.cc"
with open('rnn_model.h', 'w') as file:
  file.write(hex_to_c_array(tflite_model_hex, c_model_name))