import pickle

fingerprints_db={}
with open("fingerprints_db.pkl", mode="wb") as song_file:
        pickle.dump(fingerprints_db, song_file)
    
songs_db={}
with open("songs_db.pkl", mode="wb") as song_file:
        pickle.dump(songs_db, song_file)


def save_to_db(name: str, artist: str, fingerprint: tuple):

    #loading the fingerprints db 
    with open("fingerprints_db.pkl", mode="rb") as opened_file:
       fp = pickle.load(opened_file)

    #loading the songs db 
    with open("songs_db.pkl", mode="rb") as opened_file:
       songs = pickle.load(opened_file)
    
    #generate id #
    id = len(songs)
    
    #creating song id
    song_id = str(name) + " - " + str(artist)

    #checking if song has already been added based on song id
    for s in songs.values():
        #if the song id already exists, then the function will end and return a message
        if song_id == s:
            return "This song has already been added!"
            
    #checking if song has already been added based on fingerprint
    for f in fp.values():
        #if the fingerprint already exists, then the function will end and return a message
        if f == fingerprint:
            return "This song has already been added!"
    
    #adding fingerprint into dict
    fp[id] = fingerprint

    #adding song into db
    songs[id] = song_id
    
    #saving song db
    with open("songs_db.pkl", mode="wb") as song_file:
        pickle.dump(songs, song_file)

    #saving fingerprint db
    with open("fingerprints_db.pkl", mode="wb") as fp_file:
        pickle.dump(fp, fp_file)

    #returning the fingerprints dict
    return fp
