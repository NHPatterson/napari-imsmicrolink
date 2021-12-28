# import os
# from pathlib import Path
# from napari_imsmicrolink._dock_widget import IMSMicroLink
#
#
# def test_ims_data_read(make_napari_viewer):
#     HERE = os.path.dirname(__file__)
#     data_fp = Path(HERE) / "data_tests" / "_test_data" / "bruker_spotlist.txt"
#     viewer = make_napari_viewer()
#     imsml = IMSMicroLink(viewer)
#     imsml.read_ims_data(data_fp)
#
#     assert imsml.ims_pixel_map
#     assert imsml.viewer.layers["IMS Pixel Map"]
#     assert imsml.viewer.layers["IMS Fiducials"]
#     assert imsml.viewer.layers["IMS ROIs"]
