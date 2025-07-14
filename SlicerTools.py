import time
import logging
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

class SlicerTools():
    def __init__(self, rpyc_conn):
        self.connector_name = "3D Slicer Tools"
        self.connector_description = "Tools for interacting with 3D  Slicer."

        self.t2v_rpyc_server = rpyc_conn

        self.tools = [
            tool(self.get_node_by_class),
            tool(self.get_visible_segments),
            tool(self.set_all_segments_visibility),
            tool(self.set_segments_visibility),
            tool(self.center_view),
        ]
        self.tool_node = ToolNode(self.tools)

    def get_node_by_class(self, class_name: str):
        """
        Return all nodes in the scene of the specified class.
        
        Args:
            class_name (str): The name of the class to get nodes for.

            class_name: vtkMRMLVolumeNode - stores a 3D image. Each voxel of a volume may be a scalar (to store images with continuous grayscale values, such as a CT image), label (to store discrete labels, such as a segmentation result), vector (for storing displacement fields or RGB color images), or tensor (MRI diffusion images). 2D image volumes are represented as single-slice 3D volumes. 4D volumes are stored in sequence nodes (vtkMRMLSequenceNode).

            class_name: vtkMRMLModelNode - stores a surface mesh (polygonal elements, points, lines, etc.) or a volumetric mesh (tetrahedral, wedge elements, unstructured grid, etc.).

            class_name: vtkMRMLSegmentationNode - complex data node that can store an image segmentation (also known as contouring, labeling). It can store multiple representations internally; for example it can store both a binary labelmap image and a closed surface mesh.

            class_name: vtkMRMLMarkupsNode and subclasses - stores simple geometrical objects, such as point lists (formerly called “fiducial lists”), lines, angles, curves, planes for annotation and measurements.

            class_name: vtkMRMLTransformNode - stores a geometrical transformation that can be applied to any transformable nodes. A transformation can contain any number of linear or non-linear (warping) transforms chained together. In general, it is recommended to use vtkMRMLTransformNode. Child types (vtkMRMLLinearTransformNode, vtkMRMLBSplineTransformNode, vtkMRMLGridTransformNode) are kept for backward compatibility and to allow filtering for specific transformation types in user interface widgets.

            class_name: vtkMRMLTextNode - stores text data, such as configuration files, descriptive text, etc.

            class_name: vtkMRMLTableNode - stores tabular data (multiple scalar or vector arrays), used mainly for showing quantitative results in tables and plots


        Returns:
            str: A formatted string listing node names, or indicating if none exist.
        """
        tic = time.time()
        info = []
        nodes = self.t2v_rpyc_server.root.slicer_util_getNodesByClass(class_name)
        if nodes:
            info.append(f"{class_name} nodes:")
            for node in nodes:
                info.append(f"- {node}")
        else:
            info.append(f"No {class_name} nodes found in scene")

        toc = time.time()
        logging.info(f"get_node_by_class executed in {toc - tic:.4f} seconds")
        return "\n".join(info)

    def get_visible_segments(self, segmentation_node: str):
        """
        Context tool: Lists all currently visible segments in a specified segmentation node, including their color.

        Args:
            segmentation_node (str): The node ID of the segmentation node to inspect.

        Returns:
            str: A formatted string listing visible segments with names and RGB colors, or an empty list message.
        """
        tic = time.time()
        info = []
        segmentationNode = self.t2v_rpyc_server.root.slicer_util_getNode(segmentation_node)
        segmentationDisplayNode = segmentationNode.GetDisplayNode()
        visible_segment_ids = segmentationDisplayNode.GetVisibleSegmentIDs()

        for segmentId in visible_segment_ids:
            segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
            # Get segment display properties
            color = [0.0, 0.0, 0.0]  # Initialize RGB array
            segmentationDisplayNode.GetSegmentColor(segmentId, color)
            info.append(f"{segment.GetName()} (Color: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")

        info = f"The segmentation node {segmentation_node} contains the following visible segments: {','.join(info)}"
        toc = time.time()
        logging.info(f"get_visible_segments executed in {toc - tic:.4f} seconds")
        return info
    
    def _get_segmentation_and_display_node(self, segmentation_node_name: str):
        nodes = self.t2v_rpyc_server.root.slicer_mrmlScene_GetNodesByName(segmentation_node_name)
        if nodes.GetNumberOfItems() != 1:
            raise ValueError(f"Expected exactly one node named '{segmentation_node_name}', found {nodes.GetNumberOfItems()}")

        segmentationNode = nodes.GetItemAsObject(0)

        # Check if the node is actually a segmentation node
        if not segmentationNode.IsA('vtkMRMLSegmentationNode'):
            raise TypeError(f"Node '{segmentation_node_name}' is not a segmentation node. Found: {segmentationNode.GetClassName()}")

        displayNode = segmentationNode.GetDisplayNode()
        if not displayNode:
            raise ValueError(f"Segmentation node '{segmentation_node_name}' has no display node")

        return segmentationNode, displayNode
    
    def set_all_segments_visibility(self, segmentation_node_name: str, visible: bool):
        """
        Action tool: Sets the visibility of all segments
        (i.e. shows or hides all segments)
        within a specified segmentation node.

        Args:
            segmentation_node_name (str): The name of the segmentation node to modify.
            visible (bool): True to show all segments, False to hide all segments.
        """
        tic = time.time()
        segmentationNode, displayNode = self._get_segmentation_and_display_node(segmentation_node_name)
        wasModified = displayNode.StartModify()
        displayNode.SetAllSegmentsVisibility(visible)
        displayNode.EndModify(wasModified)
        toc = time.time()
        logging.info(f"set_all_segments_visibility executed in {toc - tic:.4f} seconds")
        return f"All segments in {segmentation_node_name} are now {'visible' if visible else 'hidden'}"
    
    def set_segments_visibility(self, segmentation_node_name: str, segment_names: list[str], visible: bool):
        """
        Action tool: Shows or hides a specific list of segments by name within a segmentation node and centers the view.

        Args:
            segmentation_node_name (str): The node name of the segmentation node to modify.
            segment_names (list[str]): A list of segment names to set visibility.
            visible (bool): True to show the specified segments, False to hide them.
        """
        tic = time.time()
        segmentationNode, displayNode = self._get_segmentation_and_display_node(segmentation_node_name)
        segmentation = segmentationNode.GetSegmentation()
        wasModified = displayNode.StartModify()
        for segment_name in segment_names:
            segment_id = segmentation.GetSegmentIdBySegmentName(segment_name)
            displayNode.SetSegmentVisibility(segment_id, visible)
        displayNode.EndModify(wasModified)
        self.center_view()
        toc = time.time()
        logging.info(f"set_segments_visibility executed in {toc - tic:.4f} seconds")
        return f"Segments {', '.join(segment_names)} in {segmentation_node_name} are now {'visible' if visible else 'hidden'}"
    
    def center_view(self) -> str:
        """
        Action tool: Centers all 2D slice views and the 3D view on the currently loaded data.

        Returns:
            str: A success or error message.
        """
        tic = time.time()
        try:
            layoutManager = self.t2v_rpyc_server.root.slicer_app_layoutManager()

            threeDWidget = layoutManager.threeDWidget(0)
            if threeDWidget:
                threeDWidget.threeDView().resetFocalPoint()

            for name in ['Red', 'Yellow', 'Green']:
                sliceWidget = layoutManager.sliceWidget(name)
                if sliceWidget:
                    sliceWidget.sliceController().fitSliceToBackground()
            toc = time.time()
            logging.info(f"center_view executed in {toc - tic:.4f} seconds")
            return "Successfully centered all views"
        except Exception as e:
            return f"Error centering views: {str(e)}"