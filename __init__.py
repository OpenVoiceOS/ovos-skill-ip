# Copyright 2017 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from shutil import which
from subprocess import check_output, CalledProcessError

from ifaddr import get_adapters
from ovos_workshop.decorators import intent_handler
from ovos_workshop.intents import IntentBuilder
from ovos_workshop.skills import OVOSSkill


def get_ifaces(ignore_list=None):
    """ Build a dict with device names and their associated ip address.

    Arguments:
        ignore_list(list): list of devices to ignore. Defaults to "lo"

    Returns:
        (dict) with device names as keys and ip addresses as value.
    """
    ignore_list = ignore_list or ['lo']
    res = {}
    for iface in get_adapters():
        # ignore "lo" (the local loopback)
        if iface.ips and iface.name not in ignore_list:
            for addr in iface.ips:
                if addr.is_IPv4:
                    res[iface.nice_name] = addr.ip
                    break
    return res


class IPSkill(OVOSSkill):

    def initialize(self):
        # Only register the SSID intent if iwlist is installed on the system
        if which("iwlist"):  # TODO - use bus events to get this info, not iwlist
            self.register_intent_file("what.ssid.intent",
                                      self.handle_SSID_query)

    @intent_handler(IntentBuilder("IPIntent").require("query").require("ip"))
    def handle_query_IP(self, message):
        addr = get_ifaces()
        dot = self.dialog_renderer.render("dot")

        if len(addr) == 0:
            self.speak_dialog("no_network_connection")
            return
        elif len(addr) == 1:
            self.enclosure.deactivate_mouth_events()
            iface, ip = addr.popitem()
            self.gui_show(ip)
            ip_spoken = ip.replace(".", " " + dot + " ")
            self.speak_dialog("my_address_is",
                              {'ip': ip_spoken}, wait=True)
        else:
            self.enclosure.deactivate_mouth_events()
            for iface in addr:
                ip = addr[iface]
                self.gui_show(ip)
                ip_spoken = ip.replace(".", " " + dot + " ")
                self.speak_dialog("my_address_on_x_is_y",
                                  {'interface': iface, 'ip': ip_spoken},
                                  wait=True)

        self.enclosure.activate_mouth_events()
        self.enclosure.mouth_reset()

    def handle_SSID_query(self, message):
        addr = get_ifaces()
        ssid = None
        if len(addr) == 0:
            self.speak_dialog("no_network_connection")
            return

        # TODO - use bus api for PHAL network manager that reports this instead
        try:
            scanoutput = check_output(["iwlist", "wlan0", "scan"])

            for line in scanoutput.split():
                line = line.decode("utf-8")
                if line[:5] == "ESSID":
                    ssid = line.split('"')[1]
        except CalledProcessError:
            # Computer has no wlan0
            pass
        finally:
            if ssid:
                self.speak(ssid)
            else:
                self.speak_dialog("ethernet.connection")

    @intent_handler(IntentBuilder("LastIPDigitsIntent").require("query").require("ip")
                    .require("last").optionally("digits"))
    def handle_query_last_part_IP(self, message):
        ip = None
        addr = get_ifaces()
        if len(addr) == 0:
            self.speak_dialog("no_network_connection")
            return

        self.enclosure.deactivate_mouth_events()
        if "wlan0" in addr:
            # Wifi is probably the one we're looking for
            ip = addr['wlan0']
        elif "eth0" in addr:
            # If there's no wifi report the eth0
            ip = addr['eth0']
        elif len(addr) == 1:
            # If none of the above matches and there's only one device
            ip = list(addr.values())[0]

        if ip:
            self.gui_show(ip)
            self.speak_last_digits(ip)
        else:
            # Ok now I don't know, I'll just report them all
            self.speak_multiple_last_digits(addr)

        self.enclosure.activate_mouth_events()
        self.enclosure.mouth_reset()

    def gui_show(self, ip):
        self.gui.show_text(ip)
        self.enclosure.mouth_text(ip)

    def speak_last_digits(self, ip):
        ip_end = ip.split(".")[-1]
        self.gui_show(ip_end)
        self.speak_dialog("last_digits", data={"digits": ip_end}, wait=True)

    def speak_multiple_last_digits(self, addr):
        for key in addr:
            ip_end = addr[key].split(".")[-1]
            self.gui_show(ip_end)
            self.speak_dialog("last_digits_device",
                              data={'device': key, 'digits': ip_end},
                              wait=True)
