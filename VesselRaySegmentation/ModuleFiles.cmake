set(MODULE_NAME VesselRaySegmentation)
set(MODULE_PYTHON_SCRIPTS
    ${MODULE_NAME}.py
    VesselRaySegmentationLib/__init__.py
    VesselRaySegmentationLib/browser.py
    VesselRaySegmentationLib/centerlines.py
    VesselRaySegmentationLib/logic.py
    VesselRaySegmentationLib/monai/__init__.py
    VesselRaySegmentationLib/monai/client.py
    VesselRaySegmentationLib/monai/images.py
    VesselRaySegmentationLib/monai/sessions.py
    VesselRaySegmentationLib/monai/submission.py
    VesselRaySegmentationLib/scene.py
    VesselRaySegmentationLib/tasks.py
    VesselRaySegmentationLib/widgets.py)
set(MODULE_PYTHON_RESOURCES Resources/Icons/${MODULE_NAME}.png
                            VesselRaySegmentationLib/Resources/requirements.txt)
