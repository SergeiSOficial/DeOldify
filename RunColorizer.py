#NOTE:  This must be the first call in order to work properly!
from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import *
torch.backends.cudnn.benchmark=True
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
import os
import shutil
import argparse
from subprocess import call

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="./test_images/", help="Test images")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Restored images",
    )
    parser.add_argument("--GPU", type=int, default="-1", help="-1 for CPU, or 0..7 to GPU0...GPU7")
    parser.add_argument(
        "--render_factor", type=str, default="35", help="Max is 45 with 11GB video cards. 35 is a good default")
    parser.add_argument("--artistic", type=bool, default="True", help="Set artistic to False if you're having trouble getting a good render.  Chances are it will work with the Stable model.")
    
    opts = parser.parse_args()

    gpu1 = opts.GPU
    render_factor = opts.render_factor

    # resolve relative paths before changing directory
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)

    main_environment = os.getcwd()

    ## Stage 1: Overall Quality Improve
    source_path = opts.input_folder
    output_path = opts.output_folder
    if not os.path.exists(output_path):
        os.remove(output_path)
        os.makedirs(output_path)

    #choices:  CPU, GPU0...GPU7
    if gpu1 >= 0:
        device.set(device=gpu1)
    else:
        device.set(device=DeviceId.CPU)

    colorizer = get_image_colorizer(artistic=opts.artistic)

    # # Instructions
    # 
    # ### source_path
    # Name this whatever sensible image path (plus extension of jpg/png/ext) you want!  Sensible means the path exists and the file exists if source_url=None.
    # 
    # ### render_factor
    # The default value of 35 has been carefully chosen and should work -ok- for most scenarios (but probably won't be the -best-). This determines resolution at which the color portion of the image is rendered. Lower resolution will render faster, and colors also tend to look more vibrant. Older and lower quality images in particular will generally benefit by lowering the render factor. Higher render factors are often better for higher quality images, but the colors may get slightly washed out. 
    # 
    # ### result_path
    # Ditto- don't change.
    # 
    # 
    # ## Pro Tips
    # 2. Keep in mind again that you can go up top and set artistic to False for the colorizer to use the 'Stable' model instead.  This will often tend to do better on portraits, and natural landscapes.  
    # 
    # 
    # ## Troubleshooting
    # If you get a 'CUDA out of memory' error, you probably have the render_factor too high.  The max is 45 on 11GB video cards.

    # ## Colorize!!
    render_factor = 35  #@param {type: "slider", min: 7, max: 40}
    watermarked = True #@param {type:"boolean"}
    
    #NOTE:  Make source_url None to just read from file at ./video/source/[file_name] directly without modification
    entries = os.listdir(source_path)
    for entry in entries:
        source_url = os.path.join(source_path, entry)
        print("Processing file: ")
        print(source_url)
        result_path = colorizer.plot_transformed_image(source_url, render_factor=render_factor, display_render_factor=True, figsize=(8,8))
        # Copy result files        
        head_result_path = os.path.split(result_path)
        shutil.copy(result_path, output_path + '/' + head_result_path[1])
