from ome_types.model import Image, Pixels, Channel, OME
from ome_types.model.simple_types import UnitsLength

pixel_metadata = {
    "size_c": 1000,
    "size_y": 1000,
    "size_x": 1000,
    "size_z": 1,
    "size_t": 1,
    "dimension_order": "XYCZT",
    "type": "uint16",
    "physical_size_x": 0.65,
    "physical_size_y": 0.65,
}
channel_meta_list = [{"name": "t1", "SamplesPerPixel": 1, "color": "FF00FF5B"}]


def generate_ome(image_name, pixel_metadata: dict, channel_meta_list: list):
    pixel_metadata.update(
        {
            "id": "Pixels:0",
            "physical_size_x_unit": UnitsLength.MICROMETER,
            "physical_size_y_unit": UnitsLength.MICROMETER,
            "metadata_only": True,
        }
    )
    ome = OME()
    ome.images.append(
        Image(
            id="Image:0",
            name=image_name,
            pixels=Pixels(
                **pixel_metadata,
            ),
        )
    )

    for idx, ch in enumerate(channel_meta_list):
        ch.update({"id": f"Channel:{idx}"})
        ome.images[0].pixels.channels.append(Channel(**ch))

    return ome
