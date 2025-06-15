import ntcore

class Pipeline:
    def start(self):
        """Start pipeline
        """
        return

    def stop(self):
        """Stop pipeline
        """
        return

    def exit(self):
        """Exit pipeline
        """
        return

    def _update_config(self, config: dict, event: ntcore.Event) -> dict:
        """Update config for pipeline

        Args:
            config (dict): config
            event (ntcore.Event): NT update event

        Returns:
            dict: Updated config
        """

        data_type = event.data.topic.getType()
        topic_name = event.data.topic.getName().split("/")[-1]

        if topic_name not in config:
            logging.exception("Invalid config")
            return config

        match data_type:
            case ntcore.NetworkTableType.kBoolean:
                config[topic_name] = event.data.value.getBoolean()
            case ntcore.NetworkTableType.kBooleanArray:
                config[topic_name] = event.data.value.getBooleanArray()
            case ntcore.NetworkTableType.kDouble:
                config[topic_name] = event.data.value.getDouble()
            case ntcore.NetworkTableType.kDoubleArray:
                config[topic_name] = event.data.value.getDoubleArray()
            case ntcore.NetworkTableType.kInteger:
                config[topic_name] = event.data.value.getInteger()
            case ntcore.NetworkTableType.kIntegerArray:
                config[topic_name] = event.data.value.getIntegerArray()
            case ntcore.NetworkTableType.kRaw:
                config[topic_name] = event.data.value.getRaw()
            case ntcore.NetworkTableType.kString:
                config[topic_name] = event.data.value.getString()
            case ntcore.NetworkTableType.kStringArray:
                config[topic_name] = event.data.value.getStringArray()
            case _:
                logging.exception("Unsupported config data type!")

        return config

