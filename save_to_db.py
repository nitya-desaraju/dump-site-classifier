#the textbook said I would need this
import pickle

#creating the song dictionary/database
songs_db={}
with open("songs_db.pkl", mode="wb") as song_file:
        pickle.dump(songs_db, song_file)

#function takes in the song name, artist, and fingerprint
def save_to_db(name: str, artist: str, fingerprint: tuple):

    #loading the db 
    with open("songs_db.pkl", mode="rb") as opened_file:
       songs = pickle.load(opened_file)
    
    #creating song id
    song_id = str(name) + " - " + str(artist)

    #checking if song has already been added based on song id
    for key in songs:
        #if the song id already exists, then the function will end and return a message
        if song_id == key:
            return "This song has already been added!"
            
    #checking if song has already been added based on fingerprint
    for f in songs.values():
        #if the fingerprint already exists, then the function will end and return a message
        if f == fingerprint:
            return "This song has already been added!"
            
    
    #adding song into dict
    songs[song_id] = fingerprint 
    
    #saving the dict -> am I supposed to specify where this goes? is it just going to save?
    with open("songs_db.pkl", mode="wb") as song_file:
        pickle.dump(songs, song_file)

    #returning the dict
    return songs


