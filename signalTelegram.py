import asyncio
import telegram
def sendMessage():
    asyncio.run(packMessage())

async def packMessage():
    bot = telegram.Bot(token='6770652347:AAGumoHubw4hnC6mvrlblqA3l24WSuC8i-A')
    await bot.sendMessage(chat_id="-4144201734", text="test")