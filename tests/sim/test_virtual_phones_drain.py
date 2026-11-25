"""Virtual phones must keep reading server pushes, or the socket stops answering pings (soak finding)."""

import asyncio

import websockets

from sim.room import DEFAULT_ROOM, SimPhone, simulate
from sim.virtual_phones import stream


def test_virtual_phone_answers_pings_after_many_server_pushes():
    rec = simulate(DEFAULT_ROOM, [SimPhone("p1", 1.0, 1.0)], [], 2.0)
    pong: list[bool] = []

    async def handler(ws):
        await ws.recv()  # phone_hello
        for _ in range(40):  # far more than the client's default max_queue of 16
            await ws.send('{"type": "phone_status"}')
        await asyncio.sleep(0.3)
        try:
            await asyncio.wait_for(await ws.ping(), 2.0)
            pong.append(True)
        except TimeoutError:
            pong.append(False)
        async for _ in ws:  # keep accepting audio until the phone hangs up
            pass

    async def main():
        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            await stream(f"ws://127.0.0.1:{port}", rec)

    asyncio.run(main())
    assert pong == [True]
