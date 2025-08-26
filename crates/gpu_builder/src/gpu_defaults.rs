#[doc(hidden)]
pub mod hidden {
    use crate::{BuildResultType, Builder};
    use core::marker::PhantomData;
    use core::mem::MaybeUninit;
    use core::num::*;

    macro_rules! impl_builder {
    ($($t:ident)*) => {
        $(
            impl<'a> Builder<'a> for $t {
                type Output = $t;
            }
            impl BuildResultType for $t {}
        )*
    }
}

    impl_builder!(
        usize u8 u16 u32 u64 u128
        isize i8 i16 i32 i64 i128
        f32 f64
        bool char

        NonZeroU8 NonZeroU16 NonZeroU32 NonZeroU64 NonZeroU128
    );

    impl<'a, T: Builder<'a>> Builder<'a> for *const T {
        type Output = *const T::Output;
    }
    //just implement BuildResultType for all to overdo the constraint
    impl<'a, T: 'static> BuildResultType for *const T {}
    impl<'a, T: Builder<'a>> Builder<'a> for *mut T {
        type Output = *mut T::Output;
    }
    //just implement BuildResultType for all to overdo the constraint
    impl<'a, T: 'static> BuildResultType for *mut T {}
    impl<'a, T: Builder<'a, Output = T> + 'static> Builder<'a> for MaybeUninit<T> {
        type Output = MaybeUninit<T::Output>;
    }
    impl<'a, T: Builder<'a, Output = T> + 'static> BuildResultType for MaybeUninit<T> {}
    impl<'a, T: Builder<'a, Output = T> + 'static> Builder<'a> for Option<T> {
        type Output = Option<T::Output>;
    }
    impl<'a, T: Builder<'a, Output = T> + 'static> BuildResultType for Option<T> {}
    impl<'a, L: Builder<'a, Output = L> + 'static, R: Builder<'a, Output = R> + 'static> Builder<'a>
        for Result<L, R>
    {
        type Output = Result<L::Output, R::Output>;
    }
    impl<'a, L: Builder<'a, Output = L> + 'static, R: Builder<'a, Output = R> + 'static>
        BuildResultType for Result<L, R>
    {
    }
    impl<'a, T: Builder<'a, Output = T> + 'static> Builder<'a> for PhantomData<T> {
        type Output = PhantomData<T::Output>;
    }
    impl<'a, T: Builder<'a, Output = T> + 'static> BuildResultType for PhantomData<T> {}
    impl<'a, T: Builder<'a, Output = T> + 'static> Builder<'a> for Wrapping<T> {
        type Output = Wrapping<T::Output>;
    }
    impl<'a, T: Builder<'a, Output = T> + 'static> BuildResultType for Wrapping<T> {}
    impl<'a, T: Builder<'a, Output = T> + 'static, const N: usize> Builder<'a> for [T; N] {
        type Output = [T::Output; N];
    }
    impl<'a, T: Builder<'a, Output = T> + 'static, const N: usize> BuildResultType for [T; N] {}
    impl<'a> Builder<'a> for () {
        type Output = ();
    }
    impl BuildResultType for () {}
    impl<'a, A: Builder<'a, Output = A> + 'static, B: Builder<'a, Output = B> + 'static> Builder<'a>
        for (A, B)
    {
        type Output = (A::Output, B::Output);
    }
    impl<'a, A: Builder<'a, Output = A> + 'static, B: Builder<'a, Output = B> + 'static>
        BuildResultType for (A, B)
    {
    }
    impl<
            'a,
            A: Builder<'a, Output = A> + 'static,
            B: Builder<'a, Output = B> + 'static,
            C: Builder<'a, Output = C> + 'static,
        > Builder<'a> for (A, B, C)
    {
        type Output = (A::Output, B::Output, C::Output);
    }
    impl<
            'a,
            A: Builder<'a, Output = A> + 'static,
            B: Builder<'a, Output = B> + 'static,
            C: Builder<'a, Output = C> + 'static,
        > BuildResultType for (A, B, C)
    {
    }
    impl<
            'a,
            A: Builder<'a, Output = A> + 'static,
            B: Builder<'a, Output = B> + 'static,
            C: Builder<'a, Output = C> + 'static,
            D: Builder<'a, Output = D> + 'static,
        > Builder<'a> for (A, B, C, D)
    {
        type Output = (A::Output, B::Output, C::Output, D::Output);
    }
    impl<
            'a,
            A: Builder<'a, Output = A> + 'static,
            B: Builder<'a, Output = B> + 'static,
            C: Builder<'a, Output = C> + 'static,
            D: Builder<'a, Output = D> + 'static,
        > BuildResultType for (A, B, C, D)
    {
    }
    impl<
            'a,
            A: Builder<'a, Output = A> + 'static,
            B: Builder<'a, Output = B> + 'static,
            C: Builder<'a, Output = C> + 'static,
            D: Builder<'a, Output = D> + 'static,
            E: Builder<'a, Output = E> + 'static,
        > Builder<'a> for (A, B, C, D, E)
    {
        type Output = (A::Output, B::Output, C::Output, D::Output, E::Output);
    }
    impl<
            'a,
            A: Builder<'a, Output = A> + 'static,
            B: Builder<'a, Output = B> + 'static,
            C: Builder<'a, Output = C> + 'static,
            D: Builder<'a, Output = D> + 'static,
            E: Builder<'a, Output = E> + 'static,
        > BuildResultType for (A, B, C, D, E)
    {
    }
    impl<
            'a,
            A: Builder<'a, Output = A> + 'static,
            B: Builder<'a, Output = B> + 'static,
            C: Builder<'a, Output = C> + 'static,
            D: Builder<'a, Output = D> + 'static,
            E: Builder<'a, Output = E> + 'static,
            F: Builder<'a, Output = F> + 'static,
        > Builder<'a> for (A, B, C, D, E, F)
    {
        type Output = (
            A::Output,
            B::Output,
            C::Output,
            D::Output,
            E::Output,
            F::Output,
        );
    }
    impl<
            'a,
            A: Builder<'a, Output = A> + 'static,
            B: Builder<'a, Output = B> + 'static,
            C: Builder<'a, Output = C> + 'static,
            D: Builder<'a, Output = D> + 'static,
            E: Builder<'a, Output = E> + 'static,
            F: Builder<'a, Output = F> + 'static,
        > BuildResultType for (A, B, C, D, E, F)
    {
    }
    impl<
            'a,
            A: Builder<'a, Output = A> + 'static,
            B: Builder<'a, Output = B> + 'static,
            C: Builder<'a, Output = C> + 'static,
            D: Builder<'a, Output = D> + 'static,
            E: Builder<'a, Output = E> + 'static,
            F: Builder<'a, Output = F> + 'static,
            G: Builder<'a, Output = G> + 'static,
        > Builder<'a> for (A, B, C, D, E, F, G)
    {
        type Output = (
            A::Output,
            B::Output,
            C::Output,
            D::Output,
            E::Output,
            F::Output,
            G::Output,
        );
    }
    impl<
            'a,
            A: Builder<'a, Output = A> + 'static,
            B: Builder<'a, Output = B> + 'static,
            C: Builder<'a, Output = C> + 'static,
            D: Builder<'a, Output = D> + 'static,
            E: Builder<'a, Output = E> + 'static,
            F: Builder<'a, Output = F> + 'static,
            G: Builder<'a, Output = G> + 'static,
        > BuildResultType for (A, B, C, D, E, F, G)
    {
    }
    impl<
            'a,
            A: Builder<'a, Output = A> + 'static,
            B: Builder<'a, Output = B> + 'static,
            C: Builder<'a, Output = C> + 'static,
            D: Builder<'a, Output = D> + 'static,
            E: Builder<'a, Output = E> + 'static,
            F: Builder<'a, Output = F> + 'static,
            G: Builder<'a, Output = G> + 'static,
            H: Builder<'a, Output = H> + 'static,
        > Builder<'a> for (A, B, C, D, E, F, G, H)
    {
        type Output = (
            A::Output,
            B::Output,
            C::Output,
            D::Output,
            E::Output,
            F::Output,
            G::Output,
            H::Output,
        );
    }
    impl<
            'a,
            A: Builder<'a, Output = A> + 'static,
            B: Builder<'a, Output = B> + 'static,
            C: Builder<'a, Output = C> + 'static,
            D: Builder<'a, Output = D> + 'static,
            E: Builder<'a, Output = E> + 'static,
            F: Builder<'a, Output = F> + 'static,
            G: Builder<'a, Output = G> + 'static,
            H: Builder<'a, Output = H> + 'static,
        > BuildResultType for (A, B, C, D, E, F, G, H)
    {
    }
}
