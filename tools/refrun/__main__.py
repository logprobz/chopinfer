import sys

# `emit` lives outside run.py, a keyed numerics file (keys.NUMERICS), so the
# fixture assembly can change without changing any cache key.
if sys.argv[1:2] == ["emit"]:
    from tools.refrun.emit import main
    main(sys.argv[2:])
elif sys.argv[1:2] == ["preprocess"]:
    from tools.refrun.preprocessing import main
    main(sys.argv[2:])
else:
    from tools.refrun.run import main
    main()
