from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """
Space exploration has always been a fascinating topic for humanity. The idea of venturing into the unknown and discovering new worlds has captured the imagination of people for centuries. With advancements in technology, we have made significant progress in our ability to explore space. From the first moon landing to the recent Mars rover missions, we have been able to gather valuable information about our universe. However, space exploration is not without its challenges. The vast distances, harsh environments, and limited resources make it a difficult endeavor. Despite these obstacles, scientists and engineers continue to push the boundaries of what is possible in space exploration. The future holds exciting possibilities, such as manned missions to Mars and the search for extraterrestrial life. As we continue to explore space, we will undoubtedly uncover new mysteries and expand our understanding of the cosmos. 

The potential benefits of space exploration are immense, including advancements in technology, scientific discoveries, and the possibility of finding new resources. It is an endeavor that requires international collaboration and a long-term commitment. As we look to the stars, we are reminded of our shared humanity and the limitless potential of our collective efforts in exploring the final frontier. With each new discovery, we are one step closer to unlocking the secrets of the universe and understanding our place in it. The journey of space exploration is a testament to human curiosity, ingenuity, and perseverance. It is a reminder that there is always more to learn and discover, and that the pursuit of knowledge is a never-ending adventure. As we continue to push the boundaries of space exploration, we can only imagine what incredible discoveries await us in the future.

The importance of space exploration cannot be overstated. It has the potential to inspire future generations, drive technological innovation, and expand our understanding of the universe. As we continue to explore space, we must also consider the ethical implications and ensure that our efforts are sustainable and responsible. The pursuit of space exploration is a reflection of our innate desire to explore and understand the world around us, and it is a journey that will continue to captivate and inspire us for generations to come.

"""

# Initialise the splitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0
)

# Perform the split

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)