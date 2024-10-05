#!/usr/bin/env python

class MarkerHandler:
    def __init__(self):
        """
        Initializes the MarkerHandler class.
        This class is responsible for managing the observed ArUco markers and their positions.
        """
        # Dictionary to hold the positions of the markers, with the ArUco ID as the key.
        self.markers = {}
        # Dictionary to keep track of observed ArUco IDs and their corresponding indices.
        self.observed_arucos = {}

    def add_marker(self, aruco_id, position):
        """
        Adds a new ArUco marker to the handler.
        
        :param aruco_id: The ID of the ArUco marker.
        :param position: The position of the ArUco marker.
        """
        # Convert the ArUco ID to an integer.
        aruco_id = int(aruco_id)
      
        # If the ArUco marker has not been observed before, add it to the dictionaries.
        if aruco_id not in self.observed_arucos:
            # Assign an index to the new marker based on the current number of observed markers.
            index = len(self.observed_arucos)
            # Add the ArUco ID and its index to the observed_arucos dictionary.
            self.observed_arucos[aruco_id] = index
            # Add the ArUco ID and its position to the markers dictionary.
            self.markers[aruco_id] = position

    def get_index(self, aruco_id):
        """
        Retrieves the index of the given ArUco marker.
        
        :param aruco_id: The ID of the ArUco marker.
        :return: The index of the ArUco marker, or None if it is not found.
        """
        # Return the index of the ArUco marker from the observed_arucos dictionary.
        return self.observed_arucos.get(int(aruco_id), None)

    def get_marker(self, aruco_id):
        """
        Retrieves the position of the given ArUco marker.

        :param aruco_id: The ID of the ArUco marker.
        :return: The position of the ArUco marker, or None if it is not found.
        """
        # Return the position of the ArUco marker from the markers dictionary.
        return self.markers.get(int(aruco_id), None)
