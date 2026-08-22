from datasets import load_dataset

def main():
    ds = load_dataset('BeIR/scifact', 'corpus', split='corpus', streaming=True)
    for i, item in enumerate(ds):
        if i >= 3:
            break
        print(f"Corpus example {i+1}:")
        print(item)

if __name__ == '__main__':
    main()
