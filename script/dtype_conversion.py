def ip_to_int(ip):
    """
    Convert an IP address string to its integer equivalent.
    """
    try:
        return int(ipaddress.ip_address(ip))
    except ValueError:
        logger.error(f"Invalid IP address encountered: {ip}")
        return None