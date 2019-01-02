class Indexer:
    UNK_SYMBOL = "<UNK>"
    PAD_SYMBOL = "<PAD>"

    def __init__(self, obj_type=str, idx_type=int):
        self.obj2int = {}
        self.int2obj = {}

        assert obj_type != idx_type
        self.obj_type = obj_type
        self.idx_type = idx_type

    def has_idx(self, idx):
        assert type(idx) == self.idx_type
        return idx in self.int2obj

    def has_obj(self, obj):
        assert type(obj) == self.obj_type
        return obj in self.obj2int

    def get_obj(self, idx):
        if self.has_idx(idx):
            return self.int2obj[idx]
        return None

    def get_idx(self, obj):
        if self.has_obj(obj):
            return self.obj2int[obj]
        elif self.has_obj(Indexer.UNK_SYMBOL):
            return self.obj2int[Indexer.UNK_SYMBOL]
        return -1

    def add(self, obj):
        if self.has_obj(obj):
            return self.get_idx(obj)

        idx = len(self.obj2int)
        self.obj2int[obj] = idx
        self.int2obj[idx] = obj
        return idx

    @staticmethod
    def create_indexer(with_symbols=True):
        indexer = Indexer()
        if with_symbols:
            indexer.add(Indexer.PAD_SYMBOL)
            indexer.add(Indexer.UNK_SYMBOL)

        return indexer

    def __getitem__(self, item):
        assert type(item) in (self.obj_type, self.idx_type)

        if type(item) == self.obj_type:
            return self.get_idx(item)

        elif type(item) == self.idx_type:
            return self.get_obj(item)
    def __len__(self):
        assert len(self.obj2int) == len(self.int2obj)
        return len(self.obj2int)
if __name__ == '__main__':
    # Run Indexer Test
    word_indexer = Indexer.create_indexer()
    word_indexer.add("fluffy")
    word_indexer.add("bunny")

    print(word_indexer["fluffy"], word_indexer[3])
    print(word_indexer["bunny"], word_indexer[1])
    print(word_indexer["rabbit"], word_indexer[2])
