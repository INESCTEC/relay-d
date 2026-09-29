import xmltodict
from relay_d.utils.coloring_logger import logger

class DataNormalizer():

    # Initialize the data normalizer class
    def __init__(self):

        # Initialize variables
        self.robot_description = None
        self.robot_limits_map = {}

        logger.info("Initializing DataNormalizer.")

    def setup_robot_description(self, xml_string):
        try:
            self.robot_description = xmltodict.parse(xml_string)
            logger.info("Successfully parsed robot description XML.")
            self._retrieve_robot_limits()
        except Exception as e:
            logger.error(f"Error parsing robot description XML: {e}")
            raise

    def _retrieve_robot_limits(self):
        if self.robot_description is None:
            logger.error("Robot description not received yet.")
            return None, None

        try:
            joint_limits = self.robot_description['robot']['joint']

            for joint in joint_limits:
                if '@type' in joint:
                    if joint['@type'] == 'revolute' or joint['@type'] == 'prismatic':
                        limits = joint['limit']
                        self.robot_limits_map[joint['@name']] = limits
                        logger.info(f"Joint: {joint['@name']}, Limits: {limits}")
                else:
                    logger.warning(f"No limits found for joint: {joint['@name']}")

            logger.info("Successfully retrieved robot limits.")
            return joint_limits
        except KeyError as e:
            logger.error(f"Key error while retrieving robot limits: {e}")
            return None

    def is_ready(self):
        return self.robot_description is not None and self.robot_limits_map

    def has_limits(self):
        return bool(self.robot_limits_map)

    def validate_joint_limits(self):
        joint_limits = self.robot_description['robot']['joint']

        for joint in joint_limits:
            if '@type' in joint:
                if joint['@type'] == 'revolute' or joint['@type'] == 'prismatic':
                    if 'limit' not in joint:
                        logger.error(f"Joint {joint['@name']} is missing limits.")
                        return False, [joint['@name']]
            else:
                logger.warning(f"No limits found for joint: {joint['@name']}")

        logger.info("All joints have limits.")
        return True, []

    def normalize_joint_data(self, joint_data, value_type):
        if not self.robot_limits_map:
            logger.error("Robot limits not available for normalization.")
            return None

        normalized_data = {}
        for joint_name, value in joint_data.items():
            if joint_name not in self.robot_limits_map:
                logger.error(f"No limits found for joint: {joint_name}. Cannot normalize.")
                return None

            limits = self.robot_limits_map[joint_name]

            if value_type == 'position':
                min_limit = float(limits['@lower'])
                max_limit = float(limits['@upper'])
                if max_limit == min_limit:
                    logger.warning(f"Zero range for joint {joint_name}, using 0.5")
                    normalized_value = 0.5
                else:
                    normalized_value = (value - min_limit) / (max_limit - min_limit)
                normalized_data[joint_name] = normalized_value

            elif value_type == 'velocity':
                max_velocity = float(limits['@velocity'])
                if max_velocity == 0:
                    logger.warning(f"Zero max velocity for joint {joint_name}, using 0.0")
                    normalized_value = 0.0
                else:
                    normalized_value = value / max_velocity
                normalized_data[joint_name] = normalized_value

            elif value_type == 'effort':
                max_effort = float(limits['@effort'])
                if max_effort == 0:
                    logger.warning(f"Zero max effort for joint {joint_name}, using 0.0")
                    normalized_value = 0.0
                else:
                    normalized_value = value / max_effort
                normalized_data[joint_name] = normalized_value

            else:
                logger.error(f"Unknown value type: {value_type}")
                return None

        return normalized_data

