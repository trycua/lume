#!/usr/bin/env python
"""
Connection test script for Computer Server.

This script tests the WebSocket connection to the Computer Server and keeps
it alive, allowing you to verify the server is running correctly.
"""

import asyncio
import json
import websockets
import argparse
import sys


async def test_connection(host="localhost", port=8000, keep_alive=False):
    """Test connection to the Computer Server."""
    uri = f"ws://{host}:{port}/ws"
    print(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            print("Connection established!")

            # Send a test command to get screen size
            await websocket.send(json.dumps({"command": "get_screen_size", "params": {}}))
            response = await websocket.recv()
            print(f"Response: {response}")

            if keep_alive:
                print("\nKeeping connection alive. Press Ctrl+C to exit...")
                while True:
                    # Send a command every 5 seconds to keep the connection alive
                    await asyncio.sleep(5)
                    await websocket.send(
                        json.dumps({"command": "get_cursor_position", "params": {}})
                    )
                    response = await websocket.recv()
                    print(f"Cursor position: {response}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")
        return False
    except ConnectionRefusedError:
        print(f"Connection refused. Is the server running at {host}:{port}?")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Test connection to Computer Server")
    parser.add_argument("--host", default="localhost", help="Host address (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--keep-alive", action="store_true", help="Keep connection alive")
    return parser.parse_args()


async def main():
    args = parse_args()
    success = await test_connection(args.host, args.port, args.keep_alive)
    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
