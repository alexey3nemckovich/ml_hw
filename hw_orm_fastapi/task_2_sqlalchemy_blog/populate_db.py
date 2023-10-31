import asyncio
import random
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession
from database import engine
from models import Post, Comment


async def main():
    session: AsyncSession = AsyncSession(engine)

    faker = Faker()
    posts_count = 10
    comments_count = 20

    posts = []
    for i in range(posts_count):
        title = faker.sentence()
        content = faker.text()
        
        post = Post(title=title, content=content, comments=[])

        posts.append(post)
        session.add(post)
    
    for i in range(comments_count):
        post = random.choice(posts)
        content = faker.text()
        session.add(Comment(post_id=post.id, content=content, post=post))

    await session.commit()
    await session.close()


if __name__ == "__main__":
    asyncio.run(main())
